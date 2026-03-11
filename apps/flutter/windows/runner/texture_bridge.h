#ifndef RUNNER_TEXTURE_BRIDGE_H_
#define RUNNER_TEXTURE_BRIDGE_H_

#include <flutter/binary_messenger.h>
#include <flutter/encodable_value.h>
#include <flutter/method_channel.h>
#include <flutter/texture_registrar.h>

#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

class TextureBridge {
 public:
  TextureBridge(flutter::BinaryMessenger* messenger,
                flutter::TextureRegistrar* texture_registrar);
  ~TextureBridge() = default;

  TextureBridge(const TextureBridge&) = delete;
  TextureBridge& operator=(const TextureBridge&) = delete;

 private:
  class ManagedTexture {
   public:
    ManagedTexture(flutter::TextureRegistrar* texture_registrar,
                   int width,
                   int height);
    ~ManagedTexture();

    std::optional<int64_t> texture_id() const;
    bool UpdateFrame(int width, int height, const std::vector<uint8_t>& pixels);

   private:
    flutter::TextureRegistrar* texture_registrar_;
    std::unique_ptr<flutter::TextureVariant> texture_;
    std::optional<int64_t> texture_id_;

    int width_;
    int height_;

    // Keep both buffers owned for the full texture lifetime. Flutter's
    // copy callback requires returned memory to remain valid until unregister.
    std::vector<uint8_t> front_pixels_;
    std::vector<uint8_t> back_pixels_;
    bool has_pending_frame_ = false;

    FlutterDesktopPixelBuffer pixel_buffer_{};
    std::mutex mutex_;
  };

  using MethodCall = flutter::MethodCall<flutter::EncodableValue>;
  using MethodResult = flutter::MethodResult<flutter::EncodableValue>;

  void HandleMethodCall(const MethodCall& call,
                        std::unique_ptr<MethodResult> result);

  std::optional<int64_t> CreateTexture(int width, int height);
  bool UpdateTexture(int64_t texture_id,
                     int width,
                     int height,
                     const std::vector<uint8_t>& pixels);
  void DisposeTexture(int64_t texture_id);

  static const flutter::EncodableMap* ReadMap(
      const flutter::EncodableValue* value);
  static const flutter::EncodableValue* FindMapValue(
      const flutter::EncodableMap& map,
      const char* key);
  static std::optional<int64_t> ReadInt64(const flutter::EncodableValue* value);
  static std::optional<std::vector<uint8_t>> ReadBytes(
      const flutter::EncodableValue* value);

  flutter::TextureRegistrar* texture_registrar_;
  std::unique_ptr<flutter::MethodChannel<flutter::EncodableValue>>
      method_channel_;
  std::unordered_map<int64_t, std::unique_ptr<ManagedTexture>> textures_;
};

#endif  // RUNNER_TEXTURE_BRIDGE_H_
