add_executable(arda-probe
    main.c
)

target_link_libraries(arda-probe PRIVATE
    arda_shared
)

install(TARGETS arda-probe
    RUNTIME DESTINATION bin
)
