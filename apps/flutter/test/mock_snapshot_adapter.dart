import 'dart:convert';

import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart' as rust;
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart' as legacy;
import 'package:sdf_modeler_flutter/src/scene/workflow_status.dart'
    as legacy_workflow;

rust.AppSceneSnapshot parseSceneSnapshotJson(String jsonString) {
  final json = (jsonDecode(jsonString) as Map).cast<String, dynamic>();
  return _sceneSnapshot(legacy.AppSceneSnapshot.fromJson(json));
}

rust.AppWorkflowStatusSnapshot parseWorkflowStatusJson(String jsonString) {
  final json = (jsonDecode(jsonString) as Map).cast<String, dynamic>();
  return _workflowStatus(legacy_workflow.AppWorkflowStatusSnapshot.fromJson(json));
}

rust.AppVec3 _vec3(legacy.AppVec3 value) =>
    rust.AppVec3(x: value.x, y: value.y, z: value.z);

rust.AppCameraSnapshot _camera(legacy.AppCameraSnapshot value) =>
    rust.AppCameraSnapshot(
      yaw: value.yaw,
      pitch: value.pitch,
      roll: value.roll,
      distance: value.distance,
      fovDegrees: value.fovDegrees,
      orthographic: value.orthographic,
      target: _vec3(value.target),
      eye: _vec3(value.eye),
    );

rust.AppNodeSnapshot _node(legacy.AppNodeSnapshot value) => rust.AppNodeSnapshot(
  id: BigInt.from(value.id),
  name: value.name,
  kindLabel: value.kindLabel,
  visible: value.visible,
  locked: value.locked,
);

rust.AppSceneTreeNodeSnapshot _sceneTreeNode(
  legacy.AppSceneTreeNodeSnapshot value,
) => rust.AppSceneTreeNodeSnapshot(
  id: BigInt.from(value.id),
  name: value.name,
  kindLabel: value.kindLabel,
  visible: value.visible,
  locked: value.locked,
  children: value.children.map(_sceneTreeNode).toList(growable: false),
);

rust.AppSceneStatsSnapshot _sceneStats(legacy.AppSceneStatsSnapshot value) =>
    rust.AppSceneStatsSnapshot(
      totalNodes: value.totalNodes,
      visibleNodes: value.visibleNodes,
      topLevelNodes: value.topLevelNodes,
      primitiveNodes: value.primitiveNodes,
      operationNodes: value.operationNodes,
      transformNodes: value.transformNodes,
      modifierNodes: value.modifierNodes,
      sculptNodes: value.sculptNodes,
      lightNodes: value.lightNodes,
      voxelMemoryBytes: BigInt.from(value.voxelMemoryBytes),
      sdfEvalComplexity: value.sdfEvalComplexity,
      structureKey: BigInt.from(value.structureKey),
      dataFingerprint: BigInt.from(value.dataFingerprint),
      boundsMin: _vec3(value.boundsMin),
      boundsMax: _vec3(value.boundsMax),
    );

rust.AppToolSnapshot _tool(legacy.AppToolSnapshot value) => rust.AppToolSnapshot(
  activeToolLabel: value.activeToolLabel,
  shadingModeLabel: value.shadingModeLabel,
  gridEnabled: value.gridEnabled,
  manipulatorModeId: value.manipulatorModeId,
  manipulatorModeLabel: value.manipulatorModeLabel,
  manipulatorSpaceId: value.manipulatorSpaceId,
  manipulatorSpaceLabel: value.manipulatorSpaceLabel,
  manipulatorVisible: value.manipulatorVisible,
  canResetPivot: value.canResetPivot,
  pivotOffset: _vec3(value.pivotOffset),
);

rust.AppRenderOptionSnapshot _renderOption(legacy.AppRenderOptionSnapshot value) =>
    rust.AppRenderOptionSnapshot(id: value.id, label: value.label);

rust.AppRenderSettingsSnapshot _renderSettings(
  legacy.AppRenderSettingsSnapshot value,
) => rust.AppRenderSettingsSnapshot(
  shadingModes: value.shadingModes.map(_renderOption).toList(growable: false),
  shadingModeId: value.shadingModeId,
  shadingModeLabel: value.shadingModeLabel,
  showGrid: value.showGrid,
  shadowsEnabled: value.shadowsEnabled,
  shadowSteps: value.shadowSteps,
  aoEnabled: value.aoEnabled,
  aoSamples: value.aoSamples,
  aoIntensity: value.aoIntensity,
  marchMaxSteps: value.marchMaxSteps,
  sculptFastMode: value.sculptFastMode,
  autoReduceSteps: value.autoReduceSteps,
  interactionRenderScale: value.interactionRenderScale,
  restRenderScale: value.restRenderScale,
  fogEnabled: value.fogEnabled,
  fogDensity: value.fogDensity,
  bloomEnabled: value.bloomEnabled,
  bloomIntensity: value.bloomIntensity,
  gamma: value.gamma,
  tonemappingAces: value.tonemappingAces,
  crossSectionAxis: value.crossSectionAxis,
  crossSectionPosition: value.crossSectionPosition,
);

rust.AppKeyOptionSnapshot _keyOption(legacy.AppKeyOptionSnapshot value) =>
    rust.AppKeyOptionSnapshot(id: value.id, label: value.label);

rust.AppKeyComboSnapshot _keyCombo(legacy.AppKeyComboSnapshot value) =>
    rust.AppKeyComboSnapshot(
      keyId: value.keyId,
      keyLabel: value.keyLabel,
      ctrl: value.ctrl,
      shift: value.shift,
      alt: value.alt,
      shortcutLabel: value.shortcutLabel,
    );

rust.AppKeybindingSnapshot _keybinding(legacy.AppKeybindingSnapshot value) =>
    rust.AppKeybindingSnapshot(
      actionId: value.actionId,
      actionLabel: value.actionLabel,
      category: value.category,
      binding: value.binding == null ? null : _keyCombo(value.binding!),
    );

rust.AppCameraBookmarkSnapshot _cameraBookmark(
  legacy.AppCameraBookmarkSnapshot value,
) => rust.AppCameraBookmarkSnapshot(
  slotIndex: value.slotIndex,
  saved: value.saved,
);

rust.AppSettingsSnapshot _settings(legacy.AppSettingsSnapshot value) =>
    rust.AppSettingsSnapshot(
      showFpsOverlay: value.showFpsOverlay,
      showNodeLabels: value.showNodeLabels,
      showBoundingBox: value.showBoundingBox,
      showLightGizmos: value.showLightGizmos,
      autoSaveEnabled: value.autoSaveEnabled,
      autoSaveIntervalSecs: value.autoSaveIntervalSecs,
      maxExportResolution: value.maxExportResolution,
      maxSculptResolution: value.maxSculptResolution,
      cameraBookmarks: value.cameraBookmarks
          .map(_cameraBookmark)
          .toList(growable: false),
      keyOptions: value.keyOptions.map(_keyOption).toList(growable: false),
      keybindings: value.keybindings.map(_keybinding).toList(growable: false),
    );

rust.AppHistorySnapshot _history(legacy.AppHistorySnapshot value) =>
    rust.AppHistorySnapshot(canUndo: value.canUndo, canRedo: value.canRedo);

rust.AppDocumentSnapshot _document(legacy.AppDocumentSnapshot value) =>
    rust.AppDocumentSnapshot(
      currentFilePath: value.currentFilePath,
      currentFileName: value.currentFileName,
      hasUnsavedChanges: value.hasUnsavedChanges,
      recentFiles: List<String>.of(value.recentFiles, growable: false),
      recoveryAvailable: value.recoveryAvailable,
      recoverySummary: value.recoverySummary,
    );

rust.AppExportPresetSnapshot _exportPreset(legacy.AppExportPresetSnapshot value) =>
    rust.AppExportPresetSnapshot(name: value.name, resolution: value.resolution);

rust.AppExportStatusSnapshot _exportStatus(
  legacy.AppExportStatusSnapshot value,
) => rust.AppExportStatusSnapshot(
  state: value.state,
  progress: value.progress,
  total: value.total,
  resolution: value.resolution,
  phaseLabel: value.phaseLabel,
  targetFileName: value.targetFileName,
  targetFilePath: value.targetFilePath,
  formatLabel: value.formatLabel,
  message: value.message,
  isError: value.isError,
);

rust.AppExportSnapshot _export(legacy.AppExportSnapshot value) =>
    rust.AppExportSnapshot(
      resolution: value.resolution,
      minResolution: value.minResolution,
      maxResolution: value.maxResolution,
      adaptive: value.adaptive,
      presets: value.presets.map(_exportPreset).toList(growable: false),
      status: _exportStatus(value.status),
    );

rust.AppImportDialogSnapshot _importDialog(legacy.AppImportDialogSnapshot value) =>
    rust.AppImportDialogSnapshot(
      filename: value.filename,
      resolution: value.resolution,
      autoResolution: value.autoResolution,
      useAuto: value.useAuto,
      vertexCount: BigInt.from(value.vertexCount),
      triangleCount: BigInt.from(value.triangleCount),
      boundsSize: _vec3(value.boundsSize),
      minResolution: value.minResolution,
      maxResolution: value.maxResolution,
    );

rust.AppImportStatusSnapshot _importStatus(legacy.AppImportStatusSnapshot value) =>
    rust.AppImportStatusSnapshot(
      state: value.state,
      progress: value.progress,
      total: value.total,
      filename: value.filename,
      phaseLabel: value.phaseLabel,
      message: value.message,
      isError: value.isError,
    );

rust.AppImportSnapshot _import(legacy.AppImportSnapshot value) =>
    rust.AppImportSnapshot(
      dialog: value.dialog == null ? null : _importDialog(value.dialog!),
      status: _importStatus(value.status),
    );

rust.AppSculptConvertDialogSnapshot _sculptConvertDialog(
  legacy.AppSculptConvertDialogSnapshot value,
) => rust.AppSculptConvertDialogSnapshot(
  targetNodeId: BigInt.from(value.targetNodeId),
  targetName: value.targetName,
  modeId: value.modeId,
  modeLabel: value.modeLabel,
  resolution: value.resolution,
  minResolution: value.minResolution,
  maxResolution: value.maxResolution,
);

rust.AppSculptConvertStatusSnapshot _sculptConvertStatus(
  legacy.AppSculptConvertStatusSnapshot value,
) => rust.AppSculptConvertStatusSnapshot(
  state: value.state,
  progress: value.progress,
  total: value.total,
  targetName: value.targetName,
  phaseLabel: value.phaseLabel,
  message: value.message,
  isError: value.isError,
);

rust.AppSculptConvertSnapshot _sculptConvert(
  legacy.AppSculptConvertSnapshot value,
) => rust.AppSculptConvertSnapshot(
  dialog: value.dialog == null ? null : _sculptConvertDialog(value.dialog!),
  status: _sculptConvertStatus(value.status),
);

rust.AppSelectedSculptSnapshot _selectedSculpt(
  legacy.AppSelectedSculptSnapshot value,
) => rust.AppSelectedSculptSnapshot(
  nodeId: BigInt.from(value.nodeId),
  nodeName: value.nodeName,
  currentResolution: value.currentResolution,
  desiredResolution: value.desiredResolution,
);

rust.AppSculptSessionSnapshot _sculptSession(legacy.AppSculptSessionSnapshot value) =>
    rust.AppSculptSessionSnapshot(
      nodeId: BigInt.from(value.nodeId),
      nodeName: value.nodeName,
      brushModeId: value.brushModeId,
      brushModeLabel: value.brushModeLabel,
      brushRadius: value.brushRadius,
      brushStrength: value.brushStrength,
      symmetryAxisId: value.symmetryAxisId,
      symmetryAxisLabel: value.symmetryAxisLabel,
    );

rust.AppSculptSnapshot _sculpt(legacy.AppSculptSnapshot value) =>
    rust.AppSculptSnapshot(
      selected: value.selected == null ? null : _selectedSculpt(value.selected!),
      session: value.session == null ? null : _sculptSession(value.session!),
      canResumeSelected: value.canResumeSelected,
      canStop: value.canStop,
      maxResolution: value.maxResolution,
    );

rust.AppLightCookieCandidateSnapshot _lightCookieCandidate(
  legacy.AppLightCookieCandidateSnapshot value,
) => rust.AppLightCookieCandidateSnapshot(
  nodeId: BigInt.from(value.nodeId),
  name: value.name,
  kindLabel: value.kindLabel,
);

rust.AppLightPropertiesSnapshot _lightProperties(
  legacy.AppLightPropertiesSnapshot value,
) => rust.AppLightPropertiesSnapshot(
  nodeId: BigInt.from(value.nodeId),
  transformNodeId: value.transformNodeId == null
      ? null
      : BigInt.from(value.transformNodeId!),
  lightTypeId: value.lightTypeId,
  lightTypeLabel: value.lightTypeLabel,
  color: _vec3(value.color),
  intensity: value.intensity,
  range: value.range,
  spotAngle: value.spotAngle,
  castShadows: value.castShadows,
  shadowSoftness: value.shadowSoftness,
  shadowColor: _vec3(value.shadowColor),
  volumetric: value.volumetric,
  volumetricDensity: value.volumetricDensity,
  cookieNodeId: value.cookieNodeId == null
      ? null
      : BigInt.from(value.cookieNodeId!),
  cookieNodeName: value.cookieNodeName,
  cookieCandidates: value.cookieCandidates
      .map(_lightCookieCandidate)
      .toList(growable: false),
  proximityModeId: value.proximityModeId,
  proximityModeLabel: value.proximityModeLabel,
  proximityRange: value.proximityRange,
  arrayPatternId: value.arrayPatternId,
  arrayPatternLabel: value.arrayPatternLabel,
  arrayCount: value.arrayCount,
  arrayRadius: value.arrayRadius,
  arrayColorVariation: value.arrayColorVariation,
  intensityExpression: value.intensityExpression,
  intensityExpressionError: value.intensityExpressionError,
  colorHueExpression: value.colorHueExpression,
  colorHueExpressionError: value.colorHueExpressionError,
  supportsRange: value.supportsRange,
  supportsSpotAngle: value.supportsSpotAngle,
  supportsShadows: value.supportsShadows,
  supportsVolumetric: value.supportsVolumetric,
  supportsCookie: value.supportsCookie,
  supportsProximity: value.supportsProximity,
  supportsExpressions: value.supportsExpressions,
  supportsArray: value.supportsArray,
);

rust.AppLightLinkTargetSnapshot _lightLinkTarget(
  legacy.AppLightLinkTargetSnapshot value,
) => rust.AppLightLinkTargetSnapshot(
  lightNodeId: BigInt.from(value.lightNodeId),
  lightName: value.lightName,
  lightTypeLabel: value.lightTypeLabel,
  active: value.active,
  maskBit: value.maskBit,
  color: _vec3(value.color),
);

rust.AppLightLinkNodeSnapshot _lightLinkNode(
  legacy.AppLightLinkNodeSnapshot value,
) => rust.AppLightLinkNodeSnapshot(
  nodeId: BigInt.from(value.nodeId),
  nodeName: value.nodeName,
  kindLabel: value.kindLabel,
  lightMask: value.lightMask,
);

rust.AppLightLinkingSnapshot _lightLinking(legacy.AppLightLinkingSnapshot value) =>
    rust.AppLightLinkingSnapshot(
      lights: value.lights.map(_lightLinkTarget).toList(growable: false),
      geometryNodes: value.geometryNodes
          .map(_lightLinkNode)
          .toList(growable: false),
      totalVisibleLightCount: value.totalVisibleLightCount,
      maxLightCount: value.maxLightCount,
    );

rust.AppScalarPropertySnapshot _scalarProperty(
  legacy.AppScalarPropertySnapshot value,
) => rust.AppScalarPropertySnapshot(
  key: value.key,
  label: value.label,
  value: value.value,
);

rust.AppTransformPropertiesSnapshot _transformProperties(
  legacy.AppTransformPropertiesSnapshot value,
) => rust.AppTransformPropertiesSnapshot(
  positionLabel: value.positionLabel,
  position: _vec3(value.position),
  rotationDegrees: _vec3(value.rotationDegrees),
  scale: value.scale == null ? null : _vec3(value.scale!),
);

rust.AppPrimitivePropertiesSnapshot _primitiveProperties(
  legacy.AppPrimitivePropertiesSnapshot value,
) => rust.AppPrimitivePropertiesSnapshot(
  primitiveKind: value.primitiveKind,
  parameters: value.parameters.map(_scalarProperty).toList(growable: false),
);

rust.AppMaterialPropertiesSnapshot _materialProperties(
  legacy.AppMaterialPropertiesSnapshot value,
) => rust.AppMaterialPropertiesSnapshot(
  color: _vec3(value.color),
  roughness: value.roughness,
  metallic: value.metallic,
  emissive: _vec3(value.emissive),
  emissiveIntensity: value.emissiveIntensity,
  fresnel: value.fresnel,
);

rust.AppSelectedNodePropertiesSnapshot _selectedNodeProperties(
  legacy.AppSelectedNodePropertiesSnapshot value,
) => rust.AppSelectedNodePropertiesSnapshot(
  nodeId: BigInt.from(value.nodeId),
  name: value.name,
  kindLabel: value.kindLabel,
  visible: value.visible,
  locked: value.locked,
  transform: value.transform == null ? null : _transformProperties(value.transform!),
  primitive: value.primitive == null ? null : _primitiveProperties(value.primitive!),
  material: value.material == null ? null : _materialProperties(value.material!),
  light: value.light == null ? null : _lightProperties(value.light!),
);

rust.AppSceneSnapshot _sceneSnapshot(legacy.AppSceneSnapshot value) =>
    rust.AppSceneSnapshot(
      selectedNode: value.selectedNode == null ? null : _node(value.selectedNode!),
      selectedNodeProperties: value.selectedNodeProperties == null
          ? null
          : _selectedNodeProperties(value.selectedNodeProperties!),
      topLevelNodes: value.topLevelNodes.map(_node).toList(growable: false),
      sceneTreeRoots: value.sceneTreeRoots
          .map(_sceneTreeNode)
          .toList(growable: false),
      history: _history(value.history),
      document: _document(value.document),
      render: _renderSettings(value.render),
      settings: _settings(value.settings),
      export_: _export(value.export),
      import_: _import(value.import),
      sculptConvert: _sculptConvert(value.sculptConvert),
      sculpt: _sculpt(value.sculpt),
      lightLinking: _lightLinking(value.lightLinking),
      camera: _camera(value.camera),
      stats: _sceneStats(value.stats),
      tool: _tool(value.tool),
    );

rust.AppWorkflowStatusSnapshot _workflowStatus(
  legacy_workflow.AppWorkflowStatusSnapshot value,
) => rust.AppWorkflowStatusSnapshot(
  exportStatus: _exportStatus(value.exportStatus),
  importStatus: _importStatus(value.importStatus),
  sculptConvertStatus: _sculptConvertStatus(value.sculptConvertStatus),
  sceneChanged: value.sceneChanged,
);
