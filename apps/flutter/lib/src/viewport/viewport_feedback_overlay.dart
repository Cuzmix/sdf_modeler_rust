import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_feedback.dart';

class ViewportFeedbackOverlay extends StatelessWidget {
  const ViewportFeedbackOverlay({
    super.key,
    required this.feedback,
    required this.interactionPhase,
    required this.frameTimeMs,
    required this.framesPerSecond,
    required this.droppedFrameCount,
  });

  static const double _targetFramesPerSecond = 60.0;
  static const double _warningFramesPerSecond = 45.0;

  final TextureViewportFeedback? feedback;
  final String interactionPhase;
  final double? frameTimeMs;
  final double? framesPerSecond;
  final int droppedFrameCount;

  @override
  Widget build(BuildContext context) {
    final selectedNode = feedback?.selectedNode;
    final hoveredNode = feedback?.hoveredNode;
    final shouldShowHoveredNode =
        hoveredNode != null && hoveredNode.id != selectedNode?.id;
    final shouldShowStats =
        framesPerSecond != null ||
        frameTimeMs != null ||
        droppedFrameCount > 0 ||
        interactionPhase != 'idle';

    if (selectedNode == null && !shouldShowHoveredNode && !shouldShowStats) {
      return const SizedBox.shrink();
    }

    return Padding(
      padding: const EdgeInsets.all(12),
      child: Stack(
        children: [
          Align(
            alignment: Alignment.topLeft,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                if (selectedNode != null)
                  _OverlayChip(
                    label: 'Selected',
                    value: selectedNode.name,
                    accentColor: const Color(0xFF8DE1D5),
                  ),
                if (shouldShowHoveredNode) ...[
                  const SizedBox(height: 8),
                  _OverlayChip(
                    label: 'Hover',
                    value: hoveredNode.name,
                    accentColor: const Color(0xFFF3E6A7),
                  ),
                ],
              ],
            ),
          ),
          if (shouldShowStats)
            Align(
              alignment: Alignment.topRight,
              child: _OverlayChip(
                label: interactionPhase,
                value: _buildStatsLine(),
                accentColor: _statsAccentColor(),
              ),
            ),
        ],
      ),
    );
  }

  String _buildStatsLine() {
    final stats = <String>[];
    if (framesPerSecond != null) {
      stats.add('${framesPerSecond!.toStringAsFixed(1)} FPS');
    }
    if (frameTimeMs != null) {
      stats.add('${frameTimeMs!.toStringAsFixed(1)} ms');
    }
    stats.add('Dropped $droppedFrameCount');
    return stats.join(' | ');
  }

  Color _statsAccentColor() {
    final measuredFramesPerSecond = framesPerSecond;
    if (measuredFramesPerSecond == null) {
      return const Color(0xFFE4B37B);
    }
    if (measuredFramesPerSecond >= _targetFramesPerSecond) {
      return const Color(0xFF80E6A8);
    }
    if (measuredFramesPerSecond >= _warningFramesPerSecond) {
      return const Color(0xFFF3E6A7);
    }
    return const Color(0xFFFFA18B);
  }
}

class _OverlayChip extends StatelessWidget {
  const _OverlayChip({
    required this.label,
    required this.value,
    required this.accentColor,
  });

  final String label;
  final String value;
  final Color accentColor;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xCC111111),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: accentColor.withValues(alpha: 0.7)),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        child: RichText(
          text: TextSpan(
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
              color: Colors.white,
              fontWeight: FontWeight.w500,
            ),
            children: [
              TextSpan(
                text: '$label: ',
                style: TextStyle(color: accentColor),
              ),
              TextSpan(text: value),
            ],
          ),
        ),
      ),
    );
  }
}