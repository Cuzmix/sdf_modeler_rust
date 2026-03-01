import 'package:flutter/material.dart';

/// App-wide theme configuration.
class AppTheme {
  AppTheme._();

  static ThemeData dark() {
    return ThemeData(
      colorScheme: ColorScheme.fromSeed(
        seedColor: Colors.blueGrey,
        brightness: Brightness.dark,
      ),
      useMaterial3: true,
    );
  }
}
