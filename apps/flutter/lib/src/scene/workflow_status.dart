import 'scene_snapshot.dart';

class AppWorkflowStatusSnapshot {
  const AppWorkflowStatusSnapshot({
    required this.exportStatus,
    required this.importStatus,
    required this.sculptConvertStatus,
    required this.sceneChanged,
  });

  final AppExportStatusSnapshot exportStatus;
  final AppImportStatusSnapshot importStatus;
  final AppSculptConvertStatusSnapshot sculptConvertStatus;
  final bool sceneChanged;

  bool get hasActiveWorkflows =>
      exportStatus.isInProgress ||
      importStatus.isInProgress ||
      sculptConvertStatus.isInProgress;

  factory AppWorkflowStatusSnapshot.fromJson(Map<String, dynamic> json) {
    return AppWorkflowStatusSnapshot(
      exportStatus: AppExportStatusSnapshot.fromJson(
        json['export_status'] as Map<String, dynamic>? ?? const {},
      ),
      importStatus: AppImportStatusSnapshot.fromJson(
        json['import_status'] as Map<String, dynamic>? ?? const {},
      ),
      sculptConvertStatus: AppSculptConvertStatusSnapshot.fromJson(
        json['sculpt_convert_status'] as Map<String, dynamic>? ?? const {},
      ),
      sceneChanged: json['scene_changed'] as bool? ?? false,
    );
  }
}
