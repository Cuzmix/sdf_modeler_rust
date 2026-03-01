import 'package:flutter/material.dart';
import 'core/theme.dart';
import 'screens/editor_screen.dart';

class SdfModelerApp extends StatelessWidget {
  const SdfModelerApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SDF Modeler',
      theme: AppTheme.dark(),
      home: const EditorScreen(),
    );
  }
}
