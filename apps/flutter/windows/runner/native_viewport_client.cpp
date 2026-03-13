#include "native_viewport_client.h"

#include <algorithm>
#include <filesystem>
#include <string>

namespace {
std::wstring ResolveBridgeLibraryPath() {
  wchar_t module_path[MAX_PATH];
  const DWORD copied = GetModuleFileNameW(nullptr, module_path, MAX_PATH);
  if (copied == 0 || copied == MAX_PATH) {
    return L"sdf_modeler_bridge.dll";
  }

  std::filesystem::path executable_path(module_path);
  return (executable_path.parent_path() / L"sdf_modeler_bridge.dll").wstring();
}
}  // namespace

std::shared_ptr<NativeViewportClient> NativeViewportClient::TryCreate() {
  const std::wstring bridge_library_path = ResolveBridgeLibraryPath();
  HMODULE module = LoadLibraryW(bridge_library_path.c_str());
  if (module == nullptr) {
    return nullptr;
  }

  auto process_frame = reinterpret_cast<ProcessViewportFrameFn>(
      GetProcAddress(module, "sdf_modeler_native_process_viewport_frame"));
  auto free_frame = reinterpret_cast<FreeViewportFrameFn>(
      GetProcAddress(module, "sdf_modeler_native_free_viewport_frame"));

  if (process_frame == nullptr || free_frame == nullptr) {
    FreeLibrary(module);
    return nullptr;
  }

  return std::shared_ptr<NativeViewportClient>(
      new NativeViewportClient(module, process_frame, free_frame));
}

NativeViewportClient::NativeViewportClient(HMODULE module,
                                           ProcessViewportFrameFn process_frame,
                                           FreeViewportFrameFn free_frame)
    : module_(module), process_frame_(process_frame), free_frame_(free_frame) {}

NativeViewportClient::~NativeViewportClient() {
  if (module_ != nullptr) {
    FreeLibrary(module_);
  }
}

std::optional<NativeViewportRenderResult> NativeViewportClient::RenderFrame(
    int width,
    int height,
    float time_seconds,
    const NativeViewportCommands& commands) const {
  const NativeViewportFrame frame = process_frame_(static_cast<uint32_t>(std::max(width, 1)),
                                                   static_cast<uint32_t>(std::max(height, 1)),
                                                   time_seconds,
                                                   commands);
  if (frame.pixels_ptr == nullptr || frame.pixels_len == 0) {
    return std::nullopt;
  }

  NativeViewportRenderResult result;
  result.pixels.assign(frame.pixels_ptr, frame.pixels_ptr + frame.pixels_len);
  if (frame.feedback_ptr != nullptr && frame.feedback_len > 0) {
    result.feedback_json.assign(reinterpret_cast<const char*>(frame.feedback_ptr),
                                frame.feedback_len);
  }
  result.camera_animating = frame.camera_animating != 0;

  free_frame_(frame);
  return result;
}