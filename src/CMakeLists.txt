add_library(translator STATIC translator.cpp translator.hpp)
add_executable(translator_app main.cpp)
target_link_libraries(translator_app PRIVATE translator)
