import 'dart:ui' as ui;
import 'package:flutter/widgets.dart';

/// Pure-presentation widget: shows a rendered [ui.Image] or a loading spinner.
class ViewportDisplay extends StatelessWidget {
  final ui.Image? image;

  const ViewportDisplay({super.key, required this.image});

  @override
  Widget build(BuildContext context) {
    if (image != null) {
      return RawImage(
        image: image,
        fit: BoxFit.contain,
        width: double.infinity,
        height: double.infinity,
      );
    }
    return const Center(child: _LoadingIndicator());
  }
}

/// Simple loading spinner (uses Directionality-safe widgets only).
class _LoadingIndicator extends StatelessWidget {
  const _LoadingIndicator();

  @override
  Widget build(BuildContext context) {
    return const SizedBox(
      width: 48,
      height: 48,
      child: DecoratedBox(
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.fromBorderSide(
            BorderSide(color: Color(0xFF90A4AE), width: 3),
          ),
        ),
      ),
    );
  }
}
