import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';

extension AppExportStatusSnapshotBridgeX on AppExportStatusSnapshot {
  bool get isInProgress => state == 'in_progress';
}

extension AppImportStatusSnapshotBridgeX on AppImportStatusSnapshot {
  bool get isInProgress => state == 'in_progress';
}

extension AppSculptConvertStatusSnapshotBridgeX on AppSculptConvertStatusSnapshot {
  bool get isInProgress => state == 'in_progress';
}

extension AppWorkflowStatusSnapshotBridgeX on AppWorkflowStatusSnapshot {
  bool get hasActiveWorkflows =>
      exportStatus.isInProgress ||
      importStatus.isInProgress ||
      sculptConvertStatus.isInProgress;
}

AppSceneSnapshot mergeWorkflowStatusIntoScene(
  AppSceneSnapshot snapshot,
  AppWorkflowStatusSnapshot workflowStatus,
) {
  return AppSceneSnapshot(
    selectedNode: snapshot.selectedNode,
    selectedNodeProperties: snapshot.selectedNodeProperties,
    topLevelNodes: snapshot.topLevelNodes,
    sceneTreeRoots: snapshot.sceneTreeRoots,
    history: snapshot.history,
    document: snapshot.document,
    render: snapshot.render,
    settings: snapshot.settings,
    export_: AppExportSnapshot(
      resolution: snapshot.export_.resolution,
      minResolution: snapshot.export_.minResolution,
      maxResolution: snapshot.export_.maxResolution,
      adaptive: snapshot.export_.adaptive,
      presets: snapshot.export_.presets,
      status: workflowStatus.exportStatus,
    ),
    import_: AppImportSnapshot(
      dialog: snapshot.import_.dialog,
      status: workflowStatus.importStatus,
    ),
    sculptConvert: AppSculptConvertSnapshot(
      dialog: snapshot.sculptConvert.dialog,
      status: workflowStatus.sculptConvertStatus,
    ),
    sculpt: snapshot.sculpt,
    lightLinking: snapshot.lightLinking,
    viewportLights: snapshot.viewportLights,
    camera: snapshot.camera,
    stats: snapshot.stats,
    tool: snapshot.tool,
  );
}
