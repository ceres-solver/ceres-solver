APP_BUILD_SCRIPT := $(call my-dir)/Android.mk
APP_PROJECT_PATH := $(call my-dir)

APP_CPPFLAGS += -fno-exceptions
APP_CPPFLAGS += -fno-rtti

# Don't use GNU libstdc++; instead use STLPort, which is free of GPL3 issues.
APP_STL := stlport_static
APP_ABI := armeabi-v7a
