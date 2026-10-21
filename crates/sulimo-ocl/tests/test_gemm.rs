//! M3a Acceptance tests: OpenCL Q4_0 x F16 GEMM, zero-copy, and zero-spin waiting (@C:OCL-GEN9, §7 M3a).

use half::f16;
use std::fs;
use std::time::Instant;
use sulimo_core::arena::{Arena, ArenaRegion};
use sulimo_ocl::buffer::OclBuffer;
use sulimo_ocl::gemm::{gemm_q4_0_f16, gemm_q4_0_f16_scalar_reference};
use sulimo_ocl::program::OclContext;

fn read_thread_cpu_time_ticks() -> u64 {
    let tid = unsafe { libc::syscall(libc::SYS_gettid) };
    let stat_path = format!("/proc/self/task/{tid}/stat");
    let content = fs::read_to_string(stat_path).expect("read /proc/self/task/<tid>/stat");

    // stat format: pid (comm) state ... utime is field 14, stime is field 15
    let rparen = content.rfind(')').expect("closing paren in comm");
    let rest = &content[rparen + 2..];
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // In rest, field 0 is state (field 3 of full stat), utime is index 11, stime is index 12
    let utime: u64 = fields[11].parse().unwrap();
    let stime: u64 = fields[12].parse().unwrap();
    utime + stime
}

fn cosine_similarity(a: &[f16], b: &[f16]) -> f64 {
    let mut dot = 0.0f64;
    let mut norm_a = 0.0f64;
    let mut norm_b = 0.0f64;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x.to_f32() as f64;
        let yf = y.to_f32() as f64;
        dot += xf * yf;
        norm_a += xf * xf;
        norm_b += yf * yf;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn max_relative_error(gpu: &[f16], cpu: &[f16]) -> f32 {
    let mut max_err = 0.0f32;
    for (&g, &c) in gpu.iter().zip(cpu.iter()) {
        let gf = g.to_f32();
        let cf = c.to_f32();
        let abs_err = (gf - cf).abs();
        let rel_err = abs_err / cf.abs().max(1.0);
        if rel_err > max_err {
            max_err = rel_err;
        }
    }
    max_err
}

#[test]
fn test_m3a_q4_0_f16_gemm_precision_and_zero_copy() {
    let mut ctx = match OclContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("OpenCL unavailable ({e}), skipping GPU test");
            return;
        }
    };

    println!(
        "Testing on OpenCL device: {} (Driver: {}, Variant: {:?})",
        ctx.device_info.device_name, ctx.device_info.driver_version, ctx.active_variant
    );

    // Shapes: N in {8, 32, 128, 512} x (K, M)
    let test_cases = [
        (8, 256, 256),
        (32, 512, 512),
        (128, 512, 1024),
        (512, 1024, 1024),
    ];

    for &(n, k, m) in &test_cases {
        let num_blocks_per_row = k / 32;
        let total_blocks = m * num_blocks_per_row;
        let scales_bytes = total_blocks * 2;
        let quants_bytes = total_blocks * 16;
        let x_bytes = n * k * 2;
        let out_bytes = n * m * 2;

        let total_arena_bytes =
            (scales_bytes + 4096) + (quants_bytes + 4096) + (x_bytes + 4096) + (out_bytes + 4096);

        let mut arena = Arena::new(total_arena_bytes).expect("allocate arena");

        let scales_slice = arena
            .alloc(scales_bytes, ArenaRegion::Scratch)
            .expect("alloc scales");
        let quants_slice = arena
            .alloc(quants_bytes, ArenaRegion::Scratch)
            .expect("alloc quants");
        let x_slice = arena.alloc(x_bytes, ArenaRegion::Scratch).expect("alloc x");
        let out_slice = arena
            .alloc(out_bytes, ArenaRegion::Scratch)
            .expect("alloc out");

        // Fill test data with pseudo-random reproducible patterns
        let scales_f16: &mut [f16] = unsafe {
            std::slice::from_raw_parts_mut(scales_slice.as_ptr() as *mut f16, total_blocks)
        };
        for (i, item) in scales_f16.iter_mut().enumerate() {
            *item = f16::from_f32(0.05 + 0.01 * ((i % 13) as f32));
        }

        let quants_u8: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(quants_slice.as_ptr() as *mut u8, total_blocks * 16)
        };
        for (i, item) in quants_u8.iter_mut().enumerate() {
            let low = (i * 3) % 16;
            let high = (i * 7 + 1) % 16;
            *item = (low | (high << 4)) as u8;
        }

        let x_f16: &mut [f16] =
            unsafe { std::slice::from_raw_parts_mut(x_slice.as_ptr() as *mut f16, n * k) };
        for (i, item) in x_f16.iter_mut().enumerate() {
            *item = f16::from_f32(((i % 19) as f32 - 9.0) * 0.1);
        }

        // Initialize zero-copy OpenCL buffers ($ZCOPY)
        let scales_buf = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                scales_slice.as_ptr(),
                scales_slice.padded_size(),
                true,
            )
            .expect("scales zero-copy buffer")
        };

        let quants_buf = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                quants_slice.as_ptr(),
                quants_slice.padded_size(),
                true,
            )
            .expect("quants zero-copy buffer")
        };

        let x_buf = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                x_slice.as_ptr(),
                x_slice.padded_size(),
                true,
            )
            .expect("x zero-copy buffer")
        };

        let mut out_buf = unsafe {
            OclBuffer::from_host_ptr(
                &ctx.context,
                &ctx.queue,
                out_slice.as_mut_ptr(),
                out_slice.padded_size(),
                false,
            )
            .expect("out zero-copy buffer")
        };

        // Test active variant (Intel SIMD8 or Portable)
        let profile = gemm_q4_0_f16(
            &mut ctx,
            &scales_buf,
            &quants_buf,
            &x_buf,
            &mut out_buf,
            m,
            k,
            n,
            None,
        )
        .expect("gemm execution");

        // Map read to ensure coherency before reading output
        let mapped_out = out_buf.map_read(&ctx.queue).expect("map out for read");
        let gpu_out: &[f16] =
            unsafe { std::slice::from_raw_parts(mapped_out as *const f16, n * m) };

        // Compute scalar reference output on CPU ($REF)
        let mut cpu_out = vec![f16::from_f32(0.0); n * m];
        gemm_q4_0_f16_scalar_reference(scales_f16, quants_u8, x_f16, m, k, n, &mut cpu_out);

        let cos = cosine_similarity(gpu_out, &cpu_out);
        let max_err = max_relative_error(gpu_out, &cpu_out);

        println!(
            "[{}] Shape N={n}, K={k}, M={m} -> GFLOPS: {:.2}, MaxRelErr: {:.4e}, Cosine: {:.6}",
            profile.kernel_variant, profile.gflops, max_err, cos
        );

        out_buf.unmap(&ctx.queue, mapped_out).expect("unmap out");

        assert!(
            max_err <= 2e-3,
            "max relative error {max_err} exceeds threshold 2e-3"
        );
        assert!(
            cos >= 0.9999,
            "cosine similarity {cos} is below threshold 0.9999"
        );
    }
}

#[test]
fn test_m3a_zero_spin_waiter_proof() {
    let mut ctx = match OclContext::new() {
        Ok(c) => c,
        Err(_) => return,
    };

    // Long GEMM to measure CPU utilization accurately
    let n = 512;
    let k = 2048;
    let m = 2048;

    let num_blocks_per_row = k / 32;
    let total_blocks = m * num_blocks_per_row;
    let scales_bytes = total_blocks * 2;
    let quants_bytes = total_blocks * 16;
    let x_bytes = n * k * 2;
    let out_bytes = n * m * 2;

    let total_arena_bytes =
        (scales_bytes + 4096) + (quants_bytes + 4096) + (x_bytes + 4096) + (out_bytes + 4096);

    let mut arena = Arena::new(total_arena_bytes).expect("allocate arena");

    let scales_slice = arena
        .alloc(scales_bytes, ArenaRegion::Scratch)
        .expect("alloc scales");
    let quants_slice = arena
        .alloc(quants_bytes, ArenaRegion::Scratch)
        .expect("alloc quants");
    let x_slice = arena.alloc(x_bytes, ArenaRegion::Scratch).expect("alloc x");
    let out_slice = arena
        .alloc(out_bytes, ArenaRegion::Scratch)
        .expect("alloc out");

    let scales_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            scales_slice.as_ptr(),
            scales_slice.padded_size(),
            true,
        )
        .expect("scales zero-copy buffer")
    };

    let quants_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            quants_slice.as_ptr(),
            quants_slice.padded_size(),
            true,
        )
        .expect("quants zero-copy buffer")
    };

    let x_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            x_slice.as_ptr(),
            x_slice.padded_size(),
            true,
        )
        .expect("x zero-copy buffer")
    };

    let mut out_buf = unsafe {
        OclBuffer::from_host_ptr(
            &ctx.context,
            &ctx.queue,
            out_slice.as_mut_ptr(),
            out_slice.padded_size(),
            false,
        )
        .expect("out zero-copy buffer")
    };

    // Warm up one iteration
    let _ = gemm_q4_0_f16(
        &mut ctx,
        &scales_buf,
        &quants_buf,
        &x_buf,
        &mut out_buf,
        m,
        k,
        n,
        None,
    );

    // Measure CPU ticks vs wall time over 10 consecutive GEMM runs
    let start_ticks = read_thread_cpu_time_ticks();
    let start_wall = Instant::now();

    let runs = 10;
    for _ in 0..runs {
        let _ = gemm_q4_0_f16(
            &mut ctx,
            &scales_buf,
            &quants_buf,
            &x_buf,
            &mut out_buf,
            m,
            k,
            n,
            None,
        )
        .expect("gemm run");
    }

    let end_wall = start_wall.elapsed().as_secs_f64();
    let end_ticks = read_thread_cpu_time_ticks();

    let clk_tck = unsafe { libc::sysconf(libc::_SC_CLK_TCK) } as f64;
    let cpu_seconds = (end_ticks - start_ticks) as f64 / clk_tck;
    let cpu_utilization_pct = (cpu_seconds / end_wall) * 100.0;

    println!(
        "Zero-spin waiter proof: {runs} GEMM runs took {:.3}s wall, {:.4}s CPU ({:.2}% CPU utilization)",
        end_wall, cpu_seconds, cpu_utilization_pct
    );

    assert!(
        cpu_utilization_pct <= 2.0,
        "CPU utilization {:.2}% exceeded threshold of 2.0% ($BATT zero-spin requirement)",
        cpu_utilization_pct
    );
}
