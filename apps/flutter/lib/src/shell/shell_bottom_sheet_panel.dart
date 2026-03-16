import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

typedef ShellBottomSheetBuilder =
    Widget Function(BuildContext context, ScrollController scrollController);

class ShellBottomSheetPanel extends StatelessWidget {
  const ShellBottomSheetPanel({
    super.key,
    required this.builder,
  });

  final ShellBottomSheetBuilder builder;

  @override
  Widget build(BuildContext context) {
    return DraggableScrollableSheet(
      expand: false,
      minChildSize: ShellPanelPlacement.tabletSheetMinSize,
      initialChildSize: ShellPanelPlacement.tabletSheetInitialSize,
      maxChildSize: ShellPanelPlacement.tabletSheetMaxSize,
      snap: true,
      snapSizes: const [
        ShellPanelPlacement.tabletSheetInitialSize,
        ShellPanelPlacement.tabletSheetMaxSize,
      ],
      builder: (context, scrollController) {
        return ShellPanelSurface(
          padding: const EdgeInsets.fromLTRB(
            ShellTokens.panelPadding,
            ShellTokens.controlGap,
            ShellTokens.panelPadding,
            ShellTokens.panelPadding,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const _SheetHandle(),
              const SizedBox(height: ShellTokens.controlGap),
              Expanded(
                child: builder(context, scrollController),
              ),
            ],
          ),
        );
      },
    );
  }
}

class _SheetHandle extends StatelessWidget {
  const _SheetHandle();

  @override
  Widget build(BuildContext context) {
    final shellPalette = context.shellPalette;

    return Center(
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: shellPalette.handle,
          borderRadius: BorderRadius.circular(999),
          boxShadow: <BoxShadow>[
            BoxShadow(
              color: shellPalette.panelShadow.withValues(alpha: 0.3),
              blurRadius: 8,
              offset: const Offset(0, 2),
            ),
          ],
        ),
        child: const SizedBox(
          width: ShellTokens.sheetHandleWidth,
          height: ShellTokens.sheetHandleHeight,
        ),
      ),
    );
  }
}
