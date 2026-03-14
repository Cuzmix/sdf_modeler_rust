import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_bottom_sheet_panel.dart';

class ShellStackedPaneLayout extends StatelessWidget {
  const ShellStackedPaneLayout({
    super.key,
    required this.viewport,
    required this.bottomSheetBuilder,
    this.modalPanel,
  });

  final Widget viewport;
  final ShellBottomSheetBuilder bottomSheetBuilder;
  final Widget? modalPanel;

  @override
  Widget build(BuildContext context) {
    return Stack(
      fit: StackFit.expand,
      children: [
        viewport,
        Positioned.fill(
          child: ShellBottomSheetPanel(builder: bottomSheetBuilder),
        ),
        if (modalPanel != null) Positioned.fill(child: modalPanel!),
      ],
    );
  }
}
