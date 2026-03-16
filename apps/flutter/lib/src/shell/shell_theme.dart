import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';

@immutable
class ShellPalette extends ThemeExtension<ShellPalette> {
  const ShellPalette({
    required this.canvasBase,
    required this.canvasAccent,
    required this.panelBase,
    required this.panelRaised,
    required this.panelInset,
    required this.panelBorder,
    required this.panelBorderStrong,
    required this.panelShadow,
    required this.overlayScrim,
    required this.overlaySurface,
    required this.overlayText,
    required this.overlayMutedText,
    required this.overlayBorder,
    required this.viewportFrame,
    required this.viewportFrameBorder,
    required this.handle,
    required this.infoAccent,
    required this.successAccent,
    required this.warningAccent,
    required this.dangerAccent,
    required this.selectionFill,
    required this.selectionBorder,
  });

  static const ShellPalette dusk = ShellPalette(
    canvasBase: Color(0xFF0D1319),
    canvasAccent: Color(0xFF18303A),
    panelBase: Color(0xFF16212B),
    panelRaised: Color(0xFF1E2C38),
    panelInset: Color(0xFF111A22),
    panelBorder: Color(0xFF2E4353),
    panelBorderStrong: Color(0xFF506D7A),
    panelShadow: Color(0x52000000),
    overlayScrim: Color(0xA60B1015),
    overlaySurface: Color(0xD918212B),
    overlayText: Color(0xFFF0F5FA),
    overlayMutedText: Color(0xFFB5C2CE),
    overlayBorder: Color(0xFF5B7E89),
    viewportFrame: Color(0xFF070B0F),
    viewportFrameBorder: Color(0xFF314855),
    handle: Color(0xFF607785),
    infoAccent: Color(0xFF6DD4C6),
    successAccent: Color(0xFF84D8A2),
    warningAccent: Color(0xFFF2CF7E),
    dangerAccent: Color(0xFFFF9B86),
    selectionFill: Color(0xFF1C3943),
    selectionBorder: Color(0xFF79D7CB),
  );

  final Color canvasBase;
  final Color canvasAccent;
  final Color panelBase;
  final Color panelRaised;
  final Color panelInset;
  final Color panelBorder;
  final Color panelBorderStrong;
  final Color panelShadow;
  final Color overlayScrim;
  final Color overlaySurface;
  final Color overlayText;
  final Color overlayMutedText;
  final Color overlayBorder;
  final Color viewportFrame;
  final Color viewportFrameBorder;
  final Color handle;
  final Color infoAccent;
  final Color successAccent;
  final Color warningAccent;
  final Color dangerAccent;
  final Color selectionFill;
  final Color selectionBorder;

  @override
  ShellPalette copyWith({
    Color? canvasBase,
    Color? canvasAccent,
    Color? panelBase,
    Color? panelRaised,
    Color? panelInset,
    Color? panelBorder,
    Color? panelBorderStrong,
    Color? panelShadow,
    Color? overlayScrim,
    Color? overlaySurface,
    Color? overlayText,
    Color? overlayMutedText,
    Color? overlayBorder,
    Color? viewportFrame,
    Color? viewportFrameBorder,
    Color? handle,
    Color? infoAccent,
    Color? successAccent,
    Color? warningAccent,
    Color? dangerAccent,
    Color? selectionFill,
    Color? selectionBorder,
  }) {
    return ShellPalette(
      canvasBase: canvasBase ?? this.canvasBase,
      canvasAccent: canvasAccent ?? this.canvasAccent,
      panelBase: panelBase ?? this.panelBase,
      panelRaised: panelRaised ?? this.panelRaised,
      panelInset: panelInset ?? this.panelInset,
      panelBorder: panelBorder ?? this.panelBorder,
      panelBorderStrong: panelBorderStrong ?? this.panelBorderStrong,
      panelShadow: panelShadow ?? this.panelShadow,
      overlayScrim: overlayScrim ?? this.overlayScrim,
      overlaySurface: overlaySurface ?? this.overlaySurface,
      overlayText: overlayText ?? this.overlayText,
      overlayMutedText: overlayMutedText ?? this.overlayMutedText,
      overlayBorder: overlayBorder ?? this.overlayBorder,
      viewportFrame: viewportFrame ?? this.viewportFrame,
      viewportFrameBorder: viewportFrameBorder ?? this.viewportFrameBorder,
      handle: handle ?? this.handle,
      infoAccent: infoAccent ?? this.infoAccent,
      successAccent: successAccent ?? this.successAccent,
      warningAccent: warningAccent ?? this.warningAccent,
      dangerAccent: dangerAccent ?? this.dangerAccent,
      selectionFill: selectionFill ?? this.selectionFill,
      selectionBorder: selectionBorder ?? this.selectionBorder,
    );
  }

  @override
  ShellPalette lerp(ThemeExtension<ShellPalette>? other, double t) {
    if (other is! ShellPalette) {
      return this;
    }

    return ShellPalette(
      canvasBase: Color.lerp(canvasBase, other.canvasBase, t) ?? canvasBase,
      canvasAccent:
          Color.lerp(canvasAccent, other.canvasAccent, t) ?? canvasAccent,
      panelBase: Color.lerp(panelBase, other.panelBase, t) ?? panelBase,
      panelRaised: Color.lerp(panelRaised, other.panelRaised, t) ?? panelRaised,
      panelInset: Color.lerp(panelInset, other.panelInset, t) ?? panelInset,
      panelBorder: Color.lerp(panelBorder, other.panelBorder, t) ?? panelBorder,
      panelBorderStrong:
          Color.lerp(panelBorderStrong, other.panelBorderStrong, t) ??
          panelBorderStrong,
      panelShadow: Color.lerp(panelShadow, other.panelShadow, t) ?? panelShadow,
      overlayScrim:
          Color.lerp(overlayScrim, other.overlayScrim, t) ?? overlayScrim,
      overlaySurface:
          Color.lerp(overlaySurface, other.overlaySurface, t) ?? overlaySurface,
      overlayText:
          Color.lerp(overlayText, other.overlayText, t) ?? overlayText,
      overlayMutedText:
          Color.lerp(overlayMutedText, other.overlayMutedText, t) ??
          overlayMutedText,
      overlayBorder:
          Color.lerp(overlayBorder, other.overlayBorder, t) ?? overlayBorder,
      viewportFrame:
          Color.lerp(viewportFrame, other.viewportFrame, t) ?? viewportFrame,
      viewportFrameBorder:
          Color.lerp(viewportFrameBorder, other.viewportFrameBorder, t) ??
          viewportFrameBorder,
      handle: Color.lerp(handle, other.handle, t) ?? handle,
      infoAccent: Color.lerp(infoAccent, other.infoAccent, t) ?? infoAccent,
      successAccent:
          Color.lerp(successAccent, other.successAccent, t) ?? successAccent,
      warningAccent:
          Color.lerp(warningAccent, other.warningAccent, t) ?? warningAccent,
      dangerAccent:
          Color.lerp(dangerAccent, other.dangerAccent, t) ?? dangerAccent,
      selectionFill:
          Color.lerp(selectionFill, other.selectionFill, t) ?? selectionFill,
      selectionBorder:
          Color.lerp(selectionBorder, other.selectionBorder, t) ??
          selectionBorder,
    );
  }
}

extension ShellThemeContext on BuildContext {
  ShellPalette get shellPalette =>
      Theme.of(this).extension<ShellPalette>() ?? ShellPalette.dusk;
}

abstract final class ShellSurfaceStyles {
  static BorderRadius get _surfaceRadius =>
      BorderRadius.circular(ShellTokens.surfaceRadius);

  static BoxDecoration canvas(BuildContext context) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: <Color>[
          shellPalette.canvasAccent,
          shellPalette.canvasBase,
          shellPalette.panelInset,
        ],
        stops: const <double>[0.0, 0.42, 1.0],
      ),
    );
  }

  static BoxDecoration panel(BuildContext context, {bool raised = true}) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: <Color>[
          raised ? shellPalette.panelRaised : shellPalette.panelBase,
          shellPalette.panelInset,
        ],
      ),
      borderRadius: _surfaceRadius,
      border: Border.all(
        color: raised
            ? shellPalette.panelBorderStrong.withValues(alpha: 0.68)
            : shellPalette.panelBorder,
      ),
      boxShadow: <BoxShadow>[
        BoxShadow(
          color: shellPalette.panelShadow,
          blurRadius: raised ? 28 : 18,
          offset: Offset(0, raised ? 16 : 10),
        ),
      ],
    );
  }

  static BoxDecoration commandStrip(BuildContext context) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: <Color>[
          shellPalette.panelRaised,
          shellPalette.panelBase,
        ],
      ),
      borderRadius: _surfaceRadius,
      border: Border.all(color: shellPalette.panelBorderStrong),
      boxShadow: <BoxShadow>[
        BoxShadow(
          color: shellPalette.panelShadow.withValues(alpha: 0.36),
          blurRadius: 18,
          offset: const Offset(0, 10),
        ),
      ],
    );
  }

  static BoxDecoration sceneTreeTile(
    BuildContext context, {
    required bool selected,
  }) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topLeft,
        end: Alignment.bottomRight,
        colors: <Color>[
          selected
              ? shellPalette.selectionFill.withValues(alpha: 0.92)
              : shellPalette.panelBase.withValues(alpha: 0.78),
          shellPalette.panelInset.withValues(alpha: 0.96),
        ],
      ),
      borderRadius: _surfaceRadius,
      border: Border.all(
        color: selected
            ? shellPalette.selectionBorder
            : shellPalette.panelBorderStrong.withValues(alpha: 0.5),
      ),
      boxShadow: selected
          ? <BoxShadow>[
              BoxShadow(
                color: shellPalette.selectionBorder.withValues(alpha: 0.14),
                blurRadius: 18,
                offset: const Offset(0, 8),
              ),
            ]
          : const <BoxShadow>[],
    );
  }

  static BoxDecoration overlayPanel(
    BuildContext context, {
    required Color accentColor,
    bool pill = false,
  }) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      color: shellPalette.overlaySurface,
      borderRadius: BorderRadius.circular(
        pill ? 999 : ShellTokens.surfaceRadius,
      ),
      border: Border.all(color: accentColor.withValues(alpha: 0.84)),
      boxShadow: <BoxShadow>[
        BoxShadow(
          color: shellPalette.panelShadow.withValues(alpha: 0.42),
          blurRadius: 22,
          offset: const Offset(0, 12),
        ),
      ],
    );
  }

  static BoxDecoration viewportFrame(BuildContext context) {
    final shellPalette = context.shellPalette;

    return BoxDecoration(
      gradient: LinearGradient(
        begin: Alignment.topCenter,
        end: Alignment.bottomCenter,
        colors: <Color>[
          shellPalette.viewportFrameBorder.withValues(alpha: 0.3),
          shellPalette.viewportFrame,
        ],
      ),
      borderRadius: _surfaceRadius,
      border: Border.all(color: shellPalette.viewportFrameBorder),
      boxShadow: <BoxShadow>[
        BoxShadow(
          color: shellPalette.panelShadow,
          blurRadius: 28,
          offset: const Offset(0, 16),
        ),
      ],
    );
  }

  static BoxDecoration messageCard(
    BuildContext context, {
    required bool isError,
  }) {
    final theme = Theme.of(context);

    return BoxDecoration(
      color: isError
          ? theme.colorScheme.errorContainer
          : theme.colorScheme.secondaryContainer,
      borderRadius: _surfaceRadius,
      border: Border.all(
        color: isError
            ? theme.colorScheme.error.withValues(alpha: 0.55)
            : theme.colorScheme.secondary.withValues(alpha: 0.4),
      ),
    );
  }
}

ThemeData buildTouchFirstShellTheme() {
  const shellPalette = ShellPalette.dusk;
  final baseColorScheme = ColorScheme.fromSeed(
    seedColor: shellPalette.infoAccent,
    brightness: Brightness.dark,
  );
  final colorScheme = baseColorScheme.copyWith(
    primary: shellPalette.infoAccent,
    onPrimary: const Color(0xFF0A171B),
    primaryContainer: shellPalette.selectionFill,
    onPrimaryContainer: const Color(0xFFDFFAF6),
    secondary: shellPalette.warningAccent,
    onSecondary: const Color(0xFF271B00),
    secondaryContainer: const Color(0xFF3A311E),
    onSecondaryContainer: const Color(0xFFFDEFC9),
    tertiary: const Color(0xFF9CB4F1),
    onTertiary: const Color(0xFF101C3D),
    tertiaryContainer: const Color(0xFF27375B),
    onTertiaryContainer: const Color(0xFFDDE6FF),
    error: shellPalette.dangerAccent,
    onError: const Color(0xFF2D110D),
    errorContainer: const Color(0xFF4E1F18),
    onErrorContainer: const Color(0xFFFFDAD4),
    surface: shellPalette.panelInset,
    onSurface: const Color(0xFFF1F5F8),
    onSurfaceVariant: const Color(0xFFA1B0BC),
    surfaceContainerLowest: const Color(0xFF091015),
    surfaceContainerLow: shellPalette.panelInset,
    surfaceContainer: shellPalette.panelBase,
    surfaceContainerHigh: shellPalette.panelRaised,
    surfaceContainerHighest: const Color(0xFF253340),
    outline: shellPalette.panelBorderStrong,
    outlineVariant: shellPalette.panelBorder,
    shadow: Colors.black,
    scrim: shellPalette.overlayScrim,
  );
  final baseTheme = ThemeData(
    colorScheme: colorScheme,
    useMaterial3: true,
    brightness: Brightness.dark,
    materialTapTargetSize: MaterialTapTargetSize.padded,
    scaffoldBackgroundColor: shellPalette.canvasBase,
  );
  final controlShape = RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
  );
  const minimumTouchTarget = Size(0, ShellTokens.minimumTouchTarget);
  const buttonPadding = EdgeInsets.symmetric(horizontal: 18, vertical: 14);
  final inputBorder = OutlineInputBorder(
    borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
    borderSide: BorderSide(color: colorScheme.outlineVariant),
  );
  final focusedInputBorder = OutlineInputBorder(
    borderRadius: BorderRadius.circular(ShellTokens.surfaceRadius),
    borderSide: BorderSide(color: colorScheme.primary, width: 1.4),
  );
  final textTheme = _buildShellTextTheme(baseTheme.textTheme, colorScheme);

  return baseTheme.copyWith(
    textTheme: textTheme,
    scaffoldBackgroundColor: shellPalette.canvasBase,
    canvasColor: shellPalette.panelBase,
    appBarTheme: AppBarTheme(
      backgroundColor: shellPalette.panelInset.withValues(alpha: 0.9),
      foregroundColor: colorScheme.onSurface,
      surfaceTintColor: Colors.transparent,
      elevation: 0,
      scrolledUnderElevation: 0,
      titleTextStyle: textTheme.titleLarge,
    ),
    filledButtonTheme: FilledButtonThemeData(
      style: FilledButton.styleFrom(
        minimumSize: minimumTouchTarget,
        padding: buttonPadding,
        shape: controlShape,
        backgroundColor: colorScheme.primary,
        foregroundColor: colorScheme.onPrimary,
      ),
    ),
    outlinedButtonTheme: OutlinedButtonThemeData(
      style: OutlinedButton.styleFrom(
        minimumSize: minimumTouchTarget,
        padding: buttonPadding,
        shape: controlShape,
        foregroundColor: colorScheme.onSurface,
        side: BorderSide(color: colorScheme.outline),
      ),
    ),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        minimumSize: minimumTouchTarget,
        padding: buttonPadding,
        shape: controlShape,
        foregroundColor: colorScheme.primary,
      ),
    ),
    iconButtonTheme: IconButtonThemeData(
      style: ButtonStyle(
        minimumSize: const WidgetStatePropertyAll(
          Size.square(ShellTokens.minimumTouchTarget),
        ),
        maximumSize: const WidgetStatePropertyAll(
          Size.square(ShellTokens.minimumTouchTarget),
        ),
        foregroundColor: WidgetStatePropertyAll(colorScheme.onSurface),
        backgroundColor: WidgetStateProperty.resolveWith<Color?>((states) {
          if (states.contains(WidgetState.pressed)) {
            return colorScheme.surfaceContainerHigh;
          }
          return Colors.transparent;
        }),
        shape: WidgetStatePropertyAll(controlShape),
      ),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: colorScheme.surfaceContainerLow,
      labelStyle: textTheme.bodyMedium?.copyWith(
        color: colorScheme.onSurfaceVariant,
      ),
      helperStyle: textTheme.bodySmall,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 18),
      border: inputBorder,
      enabledBorder: inputBorder,
      focusedBorder: focusedInputBorder,
      errorBorder: inputBorder.copyWith(
        borderSide: BorderSide(color: colorScheme.error),
      ),
      focusedErrorBorder: focusedInputBorder.copyWith(
        borderSide: BorderSide(color: colorScheme.error, width: 1.4),
      ),
    ),
    chipTheme: baseTheme.chipTheme.copyWith(
      backgroundColor: colorScheme.surfaceContainerHighest,
      selectedColor: colorScheme.secondaryContainer,
      disabledColor: colorScheme.surfaceContainerLow,
      labelStyle: textTheme.labelMedium,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      side: BorderSide(color: colorScheme.outlineVariant),
      shape: const StadiumBorder(),
    ),
    tooltipTheme: TooltipThemeData(
      decoration: BoxDecoration(
        color: shellPalette.overlaySurface,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(
          color: shellPalette.infoAccent.withValues(alpha: 0.84),
        ),
        boxShadow: <BoxShadow>[
          BoxShadow(
            color: shellPalette.panelShadow.withValues(alpha: 0.42),
            blurRadius: 16,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      textStyle: textTheme.bodySmall?.copyWith(color: shellPalette.overlayText),
    ),
    bottomSheetTheme: const BottomSheetThemeData(
      backgroundColor: Colors.transparent,
      modalBackgroundColor: Colors.transparent,
      surfaceTintColor: Colors.transparent,
    ),
    dialogTheme: DialogThemeData(
      backgroundColor: shellPalette.panelBase,
      surfaceTintColor: Colors.transparent,
      shape: controlShape,
    ),
    progressIndicatorTheme: ProgressIndicatorThemeData(
      color: colorScheme.primary,
      linearTrackColor: colorScheme.surfaceContainerHighest,
    ),
    dividerTheme: DividerThemeData(color: colorScheme.outlineVariant),
    listTileTheme: ListTileThemeData(
      iconColor: colorScheme.onSurfaceVariant,
      textColor: colorScheme.onSurface,
      shape: controlShape,
      contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
    ),
    switchTheme: SwitchThemeData(
      trackOutlineColor: WidgetStatePropertyAll(colorScheme.outlineVariant),
    ),
    cardTheme: CardThemeData(
      margin: EdgeInsets.zero,
      color: colorScheme.surfaceContainerLow,
      elevation: 0,
      shape: controlShape,
      clipBehavior: Clip.antiAlias,
      surfaceTintColor: Colors.transparent,
    ),
    extensions: const <ThemeExtension<dynamic>>[shellPalette],
  );
}

TextTheme _buildShellTextTheme(TextTheme base, ColorScheme colorScheme) {
  const List<String> displayFallback = <String>[
    'Bahnschrift',
    'Segoe UI Semibold',
    'Segoe UI',
  ];
  const List<String> bodyFallback = <String>[
    'Segoe UI Variable',
    'Segoe UI',
    'Noto Sans',
  ];
  final themed = base.apply(
    bodyColor: colorScheme.onSurface,
    displayColor: colorScheme.onSurface,
    decorationColor: colorScheme.onSurface,
  );

  return themed.copyWith(
    titleLarge: themed.titleLarge?.copyWith(
      fontFamily: 'Bahnschrift',
      fontFamilyFallback: displayFallback,
      fontWeight: FontWeight.w700,
      letterSpacing: -0.4,
    ),
    titleMedium: themed.titleMedium?.copyWith(
      fontFamily: 'Bahnschrift',
      fontFamilyFallback: displayFallback,
      fontWeight: FontWeight.w600,
      letterSpacing: -0.2,
    ),
    titleSmall: themed.titleSmall?.copyWith(
      fontFamily: 'Bahnschrift',
      fontFamilyFallback: displayFallback,
      fontWeight: FontWeight.w600,
      letterSpacing: 0.1,
    ),
    bodyLarge: themed.bodyLarge?.copyWith(
      fontFamily: 'Segoe UI',
      fontFamilyFallback: bodyFallback,
      height: 1.35,
    ),
    bodyMedium: themed.bodyMedium?.copyWith(
      fontFamily: 'Segoe UI',
      fontFamilyFallback: bodyFallback,
      height: 1.35,
    ),
    bodySmall: themed.bodySmall?.copyWith(
      fontFamily: 'Segoe UI',
      fontFamilyFallback: bodyFallback,
      color: colorScheme.onSurfaceVariant,
      height: 1.3,
    ),
    labelLarge: themed.labelLarge?.copyWith(
      fontFamily: 'Bahnschrift',
      fontFamilyFallback: displayFallback,
      fontWeight: FontWeight.w700,
      letterSpacing: 0.35,
    ),
    labelMedium: themed.labelMedium?.copyWith(
      fontFamily: 'Segoe UI',
      fontFamilyFallback: bodyFallback,
      fontWeight: FontWeight.w600,
      letterSpacing: 0.25,
    ),
  );
}
