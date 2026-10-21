//! Conversational chat view with frame-budget token streaming and intelligent auto-scroll (§7 M5).

use crate::frame_coalescer::FrameCoalescer;
use gtk4::glib;
use gtk4::pango;
use gtk4::prelude::*;
use gtk4::{
    Align, Box as GtkBox, Button, Label, Orientation, PolicyType, ScrolledWindow, TextBuffer,
    TextView, WrapMode,
};
use std::cell::RefCell;
use std::rc::Rc;

pub struct ChatView {
    pub container: GtkBox,
    pub scrolled_window: ScrolledWindow,
    pub messages_box: GtkBox,
    pub input_text_view: TextView,
    pub input_buffer: TextBuffer,
    pub send_button: Button,
    pub stop_button: Button,
    active_assistant_label: Rc<RefCell<Option<Label>>>,
    active_assistant_text: Rc<RefCell<String>>,
    coalescer: Rc<RefCell<FrameCoalescer>>,
    user_scrolled_up: Rc<RefCell<bool>>,
}

impl ChatView {
    pub fn new() -> Self {
        let container = GtkBox::new(Orientation::Vertical, 0);
        container.set_vexpand(true);
        container.set_hexpand(true);

        // 1. Scrolled Window with Messages
        let scrolled_window = ScrolledWindow::builder()
            .hscrollbar_policy(PolicyType::Never)
            .vscrollbar_policy(PolicyType::Automatic)
            .vexpand(true)
            .hexpand(true)
            .build();

        let messages_box = GtkBox::new(Orientation::Vertical, 16);
        messages_box.set_margin_start(24);
        messages_box.set_margin_end(24);
        messages_box.set_margin_top(20);
        messages_box.set_margin_bottom(20);

        scrolled_window.set_child(Some(&messages_box));

        // Auto-scroll logic: detect if user scrolled up
        let user_scrolled_up = Rc::new(RefCell::new(false));
        let user_scrolled_clone = Rc::clone(&user_scrolled_up);

        let vadjustment = scrolled_window.vadjustment();
        vadjustment.connect_value_changed(move |adj| {
            let val = adj.value();
            let upper = adj.upper();
            let page_size = adj.page_size();
            let dist_from_bottom = upper - (val + page_size);

            // If user scrolled > 30 px away from bottom, pause auto-scroll
            *user_scrolled_clone.borrow_mut() = dist_from_bottom > 30.0;
        });

        // 2. Input Box Area
        let input_container = GtkBox::new(Orientation::Horizontal, 10);
        input_container.set_margin_start(24);
        input_container.set_margin_end(24);
        input_container.set_margin_top(12);
        input_container.set_margin_bottom(12);
        input_container.add_css_class("chat-input-bar");

        let input_text_view = TextView::builder()
            .wrap_mode(WrapMode::WordChar)
            .hexpand(true)
            .accepts_tab(false)
            .left_margin(12)
            .right_margin(12)
            .top_margin(10)
            .bottom_margin(10)
            .build();
        input_text_view.set_size_request(-1, 56);
        input_text_view.add_css_class("card");
        input_text_view.add_css_class("chat-input-field");

        let input_buffer = input_text_view.buffer();

        let send_button = Button::builder()
            .icon_name("mail-send-symbolic")
            .valign(Align::End)
            .tooltip_text("Send prompt (Ctrl+Enter)")
            .build();
        send_button.add_css_class("suggested-action");
        send_button.add_css_class("pill");

        let stop_button = Button::builder()
            .icon_name("media-playback-stop-symbolic")
            .valign(Align::End)
            .tooltip_text("Stop generation")
            .visible(false)
            .build();
        stop_button.add_css_class("destructive-action");
        stop_button.add_css_class("pill");

        input_container.append(&input_text_view);
        input_container.append(&send_button);
        input_container.append(&stop_button);

        container.append(&scrolled_window);
        container.append(&input_container);

        Self {
            container,
            scrolled_window,
            messages_box,
            input_text_view,
            input_buffer,
            send_button,
            stop_button,
            active_assistant_label: Rc::new(RefCell::new(None)),
            active_assistant_text: Rc::new(RefCell::new(String::new())),
            coalescer: Rc::new(RefCell::new(FrameCoalescer::default())),
            user_scrolled_up,
        }
    }

    /// Adds a user message bubble (right-aligned).
    pub fn add_user_message(&self, text: &str) {
        let row = GtkBox::new(Orientation::Horizontal, 0);
        row.set_halign(Align::End);

        let bubble = Label::builder()
            .label(text)
            .wrap(true)
            .wrap_mode(pango::WrapMode::WordChar)
            .xalign(0.0)
            .selectable(true)
            .build();
        bubble.add_css_class("card");
        bubble.add_css_class("user-bubble");

        row.append(&bubble);
        self.messages_box.append(&row);
        self.scroll_to_bottom();
    }

    /// Starts a new streaming assistant response bubble (left-aligned).
    pub fn start_assistant_message(&self) {
        let row = GtkBox::new(Orientation::Horizontal, 0);
        row.set_halign(Align::Start);

        let bubble = Label::builder()
            .label("")
            .wrap(true)
            .wrap_mode(pango::WrapMode::WordChar)
            .xalign(0.0)
            .selectable(true)
            .build();
        bubble.add_css_class("card");
        bubble.add_css_class("assistant-bubble");

        row.append(&bubble);
        self.messages_box.append(&row);

        *self.active_assistant_label.borrow_mut() = Some(bubble);
        self.active_assistant_text.borrow_mut().clear();
        self.coalescer.borrow_mut().reset();

        self.scroll_to_bottom();
    }

    /// Pushes a streaming token chunk to the frame coalescer.
    pub fn stream_token(&self, token: &str) {
        if let Some(flush_text) = self.coalescer.borrow_mut().push_token(token) {
            self.append_to_active_label(&flush_text);
        }
    }

    /// Flushes any pending tokens (tick callback or completion).
    pub fn flush_coalescer(&self) {
        if let Some(flush_text) = self.coalescer.borrow_mut().flush() {
            self.append_to_active_label(&flush_text);
        }
    }

    fn append_to_active_label(&self, chunk: &str) {
        let mut full_text = self.active_assistant_text.borrow_mut();
        full_text.push_str(chunk);

        if let Some(ref label) = *self.active_assistant_label.borrow() {
            label.set_text(&full_text);
        }

        // Only auto-scroll if user hasn't scrolled up
        if !*self.user_scrolled_up.borrow() {
            self.scroll_to_bottom();
        }
    }

    /// Clears the chat history.
    pub fn clear(&self) {
        while let Some(child) = self.messages_box.first_child() {
            self.messages_box.remove(&child);
        }
        *self.active_assistant_label.borrow_mut() = None;
        self.active_assistant_text.borrow_mut().clear();
        self.coalescer.borrow_mut().reset();
    }

    /// Scrolls to the very bottom of the conversation.
    pub fn scroll_to_bottom(&self) {
        let adj = self.scrolled_window.vadjustment();
        glib::idle_add_local_once(move || {
            adj.set_value(adj.upper() - adj.page_size());
        });
    }

    /// Sets generation state (toggle send / stop buttons).
    pub fn set_generating(&self, generating: bool) {
        self.send_button.set_visible(!generating);
        self.stop_button.set_visible(generating);
        self.input_text_view.set_sensitive(!generating);
    }
}

impl Default for ChatView {
    fn default() -> Self {
        Self::new()
    }
}
