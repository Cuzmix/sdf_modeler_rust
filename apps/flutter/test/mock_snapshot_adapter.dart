import 'dart:convert';

import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart'
    show Uint64List;
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart' as rust;

rust.AppSceneSnapshot parseSceneSnapshotJson(String jsonString) {
  final json = _readJsonObject(jsonDecode(jsonString));
  return _sceneSnapshot(json);
}

rust.AppWorkflowStatusSnapshot parseWorkflowStatusJson(String jsonString) {
  final json = _readJsonObject(jsonDecode(jsonString));
  return _workflowStatus(json);
}

Map<String, dynamic> _readJsonObject(Object? value) {
  if (value is Map) {
    return value.cast<String, dynamic>();
  }
  throw FormatException('Expected a JSON object.');
}

List<dynamic> _readJsonList(Object? value) {
  if (value is List) {
    return value;
  }
  return const <dynamic>[];
}

String _readString(
  Map<String, dynamic> json,
  String key, {
  String defaultValue = '',
}) {
  final value = json[key];
  return value is String ? value : defaultValue;
}

String? _readNullableString(Map<String, dynamic> json, String key) {
  final value = json[key];
  return value is String ? value : null;
}

bool _readBool(
  Map<String, dynamic> json,
  String key, {
  bool defaultValue = false,
}) {
  final value = json[key];
  return value is bool ? value : defaultValue;
}

int _readInt(Map<String, dynamic> json, String key, {int defaultValue = 0}) {
  final value = json[key];
  return value is num ? value.toInt() : defaultValue;
}

double _readDouble(
  Map<String, dynamic> json,
  String key, {
  double defaultValue = 0.0,
}) {
  final value = json[key];
  return value is num ? value.toDouble() : defaultValue;
}

BigInt _readBigInt(
  Map<String, dynamic> json,
  String key, {
  int defaultValue = 0,
}) {
  final value = json[key];
  if (value is int) {
    return BigInt.from(value);
  }
  if (value is num) {
    return BigInt.from(value.toInt());
  }
  return BigInt.from(defaultValue);
}

Map<String, dynamic>? _readOptionalObject(Map<String, dynamic> json, String key) {
  final value = json[key];
  if (value == null) {
    return null;
  }
  return _readJsonObject(value);
}

List<T> _readObjectList<T>(
  Map<String, dynamic> json,
  String key,
  T Function(Map<String, dynamic>) read,
) {
  return _readJsonList(json[key])
      .map((item) => read(_readJsonObject(item)))
      .toList(growable: false);
}

List<String> _readStringList(Map<String, dynamic> json, String key) {
  return _readJsonList(json[key])
      .whereType<String>()
      .toList(growable: false);
}

rust.AppVec3 _vec3(Map<String, dynamic> json) => rust.AppVec3(
  x: _readDouble(json, 'x'),
  y: _readDouble(json, 'y'),
  z: _readDouble(json, 'z'),
);

rust.AppCameraSnapshot _camera(Map<String, dynamic> json) =>
    rust.AppCameraSnapshot(
      yaw: _readDouble(json, 'yaw'),
      pitch: _readDouble(json, 'pitch'),
      roll: _readDouble(json, 'roll'),
      distance: _readDouble(json, 'distance'),
      fovDegrees: _readDouble(json, 'fov_degrees'),
      orthographic: _readBool(json, 'orthographic'),
      target: _vec3(_readJsonObject(json['target'])),
      eye: _vec3(_readJsonObject(json['eye'])),
    );

rust.AppNodeSnapshot _node(Map<String, dynamic> json) => rust.AppNodeSnapshot(
  id: _readBigInt(json, 'id'),
  name: _readString(json, 'name'),
  kindLabel: _readString(json, 'kind_label'),
  visible: _readBool(json, 'visible'),
  locked: _readBool(json, 'locked'),
);

rust.AppSceneTreeNodeSnapshot _sceneTreeNode(Map<String, dynamic> json) =>
    rust.AppSceneTreeNodeSnapshot(
      id: _readBigInt(json, 'id'),
      name: _readString(json, 'name'),
      kindLabel: _readString(json, 'kind_label'),
      visible: _readBool(json, 'visible'),
      locked: _readBool(json, 'locked'),
      workflowStatusId: _readString(
        json,
        'workflow_status_id',
        defaultValue: 'live',
      ),
      workflowStatusLabel: _readString(
        json,
        'workflow_status_label',
        defaultValue: 'Live',
      ),
      children: _readObjectList(json, 'children', _sceneTreeNode),
    );

rust.AppSceneTreeNodeSnapshot _sceneTreeNodeFromNode(rust.AppNodeSnapshot node) =>
    rust.AppSceneTreeNodeSnapshot(
      id: node.id,
      name: node.name,
      kindLabel: node.kindLabel,
      visible: node.visible,
      locked: node.locked,
      workflowStatusId: 'live',
      workflowStatusLabel: 'Live',
      children: const <rust.AppSceneTreeNodeSnapshot>[],
    );

rust.AppQuickActionSnapshot _quickAction(Map<String, dynamic> json) =>
    rust.AppQuickActionSnapshot(
      id: _readString(json, 'id'),
      label: _readString(json, 'label'),
      category: _readString(json, 'category'),
      enabled: _readBool(json, 'enabled', defaultValue: true),
      prominent: _readBool(json, 'prominent'),
      shortcutLabel: _readNullableString(json, 'shortcut_label'),
    );

rust.AppCommandSnapshot _command(Map<String, dynamic> json) =>
    rust.AppCommandSnapshot(
      id: _readString(json, 'id'),
      label: _readString(json, 'label'),
      category: _readString(json, 'category'),
      enabled: _readBool(json, 'enabled', defaultValue: true),
      workspaceIds: _readStringList(json, 'workspace_ids'),
      shortcutLabel: _readNullableString(json, 'shortcut_label'),
    );

rust.AppSelectionContextSnapshot _selectionContext(Map<String, dynamic> json) =>
    rust.AppSelectionContextSnapshot(
      headline: _readString(json, 'headline'),
      detail: _readString(json, 'detail'),
      selectionCount: _readInt(json, 'selection_count'),
      selectionKindId: _readString(
        json,
        'selection_kind_id',
        defaultValue: 'none',
      ),
      selectionKindLabel: _readString(
        json,
        'selection_kind_label',
        defaultValue: 'Nothing selected',
      ),
      workflowStatusId: _readString(
        json,
        'workflow_status_id',
        defaultValue: 'none',
      ),
      workflowStatusLabel: _readString(
        json,
        'workflow_status_label',
        defaultValue: 'No selection',
      ),
      quickActions: _readObjectList(json, 'quick_actions', _quickAction),
    );

rust.AppWorkspaceSnapshot _workspace(Map<String, dynamic> json) =>
    rust.AppWorkspaceSnapshot(
      id: _readString(json, 'id', defaultValue: 'blockout'),
      label: _readString(json, 'label', defaultValue: 'Blockout'),
      description: _readString(
        json,
        'description',
        defaultValue: 'Live SDF primitives, booleans, and transforms.',
      ),
    );

rust.AppSceneStatsSnapshot _sceneStats(Map<String, dynamic> json) =>
    rust.AppSceneStatsSnapshot(
      totalNodes: _readInt(json, 'total_nodes'),
      visibleNodes: _readInt(json, 'visible_nodes'),
      topLevelNodes: _readInt(json, 'top_level_nodes'),
      primitiveNodes: _readInt(json, 'primitive_nodes'),
      operationNodes: _readInt(json, 'operation_nodes'),
      transformNodes: _readInt(json, 'transform_nodes'),
      modifierNodes: _readInt(json, 'modifier_nodes'),
      sculptNodes: _readInt(json, 'sculpt_nodes'),
      lightNodes: _readInt(json, 'light_nodes'),
      voxelMemoryBytes: _readBigInt(json, 'voxel_memory_bytes'),
      sdfEvalComplexity: _readInt(json, 'sdf_eval_complexity'),
      structureKey: _readBigInt(json, 'structure_key'),
      dataFingerprint: _readBigInt(json, 'data_fingerprint'),
      boundsMin: _vec3(_readJsonObject(json['bounds_min'])),
      boundsMax: _vec3(_readJsonObject(json['bounds_max'])),
    );

rust.AppToolSnapshot _tool(Map<String, dynamic> json) => rust.AppToolSnapshot(
  activeToolLabel: _readString(json, 'active_tool_label'),
  shadingModeLabel: _readString(json, 'shading_mode_label'),
  gridEnabled: _readBool(json, 'grid_enabled'),
  manipulatorModeId: _readString(
    json,
    'manipulator_mode_id',
    defaultValue: 'translate',
  ),
  manipulatorModeLabel: _readString(
    json,
    'manipulator_mode_label',
    defaultValue: 'Move',
  ),
  manipulatorSpaceId: _readString(
    json,
    'manipulator_space_id',
    defaultValue: 'local',
  ),
  manipulatorSpaceLabel: _readString(
    json,
    'manipulator_space_label',
    defaultValue: 'Local',
  ),
  manipulatorVisible: _readBool(json, 'manipulator_visible'),
  canResetPivot: _readBool(json, 'can_reset_pivot'),
  pivotOffset: _readOptionalObject(json, 'pivot_offset') == null
      ? const rust.AppVec3(x: 0, y: 0, z: 0)
      : _vec3(_readOptionalObject(json, 'pivot_offset')!),
);

rust.AppRenderOptionSnapshot _renderOption(Map<String, dynamic> json) =>
    rust.AppRenderOptionSnapshot(
      id: _readString(json, 'id', defaultValue: 'option'),
      label: _readString(json, 'label', defaultValue: 'Option'),
    );

rust.AppRenderSettingsSnapshot _renderSettings(Map<String, dynamic> json) =>
    rust.AppRenderSettingsSnapshot(
      shadingModes: _readObjectList(json, 'shading_modes', _renderOption),
      shadingModeId: _readString(json, 'shading_mode_id', defaultValue: 'full'),
      shadingModeLabel: _readString(
        json,
        'shading_mode_label',
        defaultValue: 'Full',
      ),
      showGrid: _readBool(json, 'show_grid', defaultValue: true),
      shadowsEnabled: _readBool(json, 'shadows_enabled'),
      shadowSteps: _readInt(json, 'shadow_steps', defaultValue: 32),
      aoEnabled: _readBool(json, 'ao_enabled', defaultValue: true),
      aoSamples: _readInt(json, 'ao_samples', defaultValue: 5),
      aoIntensity: _readDouble(json, 'ao_intensity', defaultValue: 3.0),
      marchMaxSteps: _readInt(json, 'march_max_steps', defaultValue: 128),
      sculptFastMode: _readBool(json, 'sculpt_fast_mode'),
      autoReduceSteps: _readBool(json, 'auto_reduce_steps', defaultValue: true),
      interactionRenderScale: _readDouble(
        json,
        'interaction_render_scale',
        defaultValue: 0.5,
      ),
      restRenderScale: _readDouble(
        json,
        'rest_render_scale',
        defaultValue: 1.0,
      ),
      fogEnabled: _readBool(json, 'fog_enabled'),
      fogDensity: _readDouble(json, 'fog_density', defaultValue: 0.02),
      bloomEnabled: _readBool(json, 'bloom_enabled'),
      bloomIntensity: _readDouble(json, 'bloom_intensity', defaultValue: 0.3),
      gamma: _readDouble(json, 'gamma', defaultValue: 2.2),
      tonemappingAces: _readBool(json, 'tonemapping_aces'),
      crossSectionAxis: _readInt(json, 'cross_section_axis'),
      crossSectionPosition: _readDouble(json, 'cross_section_position'),
    );

rust.AppKeyOptionSnapshot _keyOption(Map<String, dynamic> json) =>
    rust.AppKeyOptionSnapshot(
      id: _readString(json, 'id', defaultValue: 'key'),
      label: _readString(json, 'label', defaultValue: 'Key'),
    );

rust.AppKeyComboSnapshot _keyCombo(Map<String, dynamic> json) =>
    rust.AppKeyComboSnapshot(
      keyId: _readString(json, 'key_id', defaultValue: 'key'),
      keyLabel: _readString(json, 'key_label', defaultValue: 'Key'),
      ctrl: _readBool(json, 'ctrl'),
      shift: _readBool(json, 'shift'),
      alt: _readBool(json, 'alt'),
      shortcutLabel: _readString(json, 'shortcut_label', defaultValue: 'Key'),
    );

rust.AppKeybindingSnapshot _keybinding(Map<String, dynamic> json) =>
    rust.AppKeybindingSnapshot(
      actionId: _readString(json, 'action_id', defaultValue: 'action'),
      actionLabel: _readString(json, 'action_label', defaultValue: 'Action'),
      category: _readString(json, 'category', defaultValue: 'General'),
      binding: _readOptionalObject(json, 'binding') == null
          ? null
          : _keyCombo(_readOptionalObject(json, 'binding')!),
    );

rust.AppCameraBookmarkSnapshot _cameraBookmark(Map<String, dynamic> json) =>
    rust.AppCameraBookmarkSnapshot(
      slotIndex: _readInt(json, 'slot_index'),
      saved: _readBool(json, 'saved'),
    );

rust.AppSettingsSnapshot _settings(Map<String, dynamic> json) =>
    rust.AppSettingsSnapshot(
      showFpsOverlay: _readBool(json, 'show_fps_overlay', defaultValue: true),
      showNodeLabels: _readBool(json, 'show_node_labels'),
      showBoundingBox: _readBool(
        json,
        'show_bounding_box',
        defaultValue: true,
      ),
      showLightGizmos: _readBool(
        json,
        'show_light_gizmos',
        defaultValue: true,
      ),
      autoSaveEnabled: _readBool(json, 'auto_save_enabled', defaultValue: true),
      autoSaveIntervalSecs: _readInt(
        json,
        'auto_save_interval_secs',
        defaultValue: 120,
      ),
      maxExportResolution: _readInt(
        json,
        'max_export_resolution',
        defaultValue: 2048,
      ),
      maxSculptResolution: _readInt(
        json,
        'max_sculpt_resolution',
        defaultValue: 320,
      ),
      cameraBookmarks: _readObjectList(
        json,
        'camera_bookmarks',
        _cameraBookmark,
      ),
      keyOptions: _readObjectList(json, 'key_options', _keyOption),
      keybindings: _readObjectList(json, 'keybindings', _keybinding),
    );

rust.AppHistorySnapshot _history(Map<String, dynamic> json) =>
    rust.AppHistorySnapshot(
      canUndo: _readBool(json, 'can_undo'),
      canRedo: _readBool(json, 'can_redo'),
    );

rust.AppDocumentSnapshot _document(Map<String, dynamic> json) =>
    rust.AppDocumentSnapshot(
      currentFilePath: _readNullableString(json, 'current_file_path'),
      currentFileName: _readNullableString(json, 'current_file_name'),
      hasUnsavedChanges: _readBool(json, 'has_unsaved_changes'),
      recentFiles: _readStringList(json, 'recent_files'),
      recoveryAvailable: _readBool(json, 'recovery_available'),
      recoverySummary: _readNullableString(json, 'recovery_summary'),
    );

rust.AppExportPresetSnapshot _exportPreset(Map<String, dynamic> json) =>
    rust.AppExportPresetSnapshot(
      name: _readString(json, 'name', defaultValue: 'Preset'),
      resolution: _readInt(json, 'resolution', defaultValue: 128),
    );

rust.AppExportStatusSnapshot _exportStatus(Map<String, dynamic> json) =>
    rust.AppExportStatusSnapshot(
      state: _readString(json, 'state', defaultValue: 'idle'),
      progress: _readInt(json, 'progress'),
      total: _readInt(json, 'total'),
      resolution: _readInt(json, 'resolution', defaultValue: 128),
      phaseLabel: _readNullableString(json, 'phase_label'),
      targetFileName: _readNullableString(json, 'target_file_name'),
      targetFilePath: _readNullableString(json, 'target_file_path'),
      formatLabel: _readNullableString(json, 'format_label'),
      message: _readNullableString(json, 'message'),
      isError: _readBool(json, 'is_error'),
    );

rust.AppExportSnapshot _exportSnapshot(Map<String, dynamic> json) =>
    rust.AppExportSnapshot(
      resolution: _readInt(json, 'resolution', defaultValue: 128),
      minResolution: _readInt(json, 'min_resolution', defaultValue: 16),
      maxResolution: _readInt(json, 'max_resolution', defaultValue: 2048),
      adaptive: _readBool(json, 'adaptive'),
      presets: _readObjectList(json, 'presets', _exportPreset),
      status: _readOptionalObject(json, 'status') == null
          ? _defaultExportSnapshot().status
          : _exportStatus(_readOptionalObject(json, 'status')!),
    );

rust.AppImportDialogSnapshot _importDialog(Map<String, dynamic> json) =>
    rust.AppImportDialogSnapshot(
      filename: _readString(json, 'filename', defaultValue: 'mesh'),
      resolution: _readInt(json, 'resolution', defaultValue: 64),
      autoResolution: _readInt(json, 'auto_resolution', defaultValue: 64),
      useAuto: _readBool(json, 'use_auto', defaultValue: true),
      vertexCount: _readBigInt(json, 'vertex_count'),
      triangleCount: _readBigInt(json, 'triangle_count'),
      boundsSize: _readOptionalObject(json, 'bounds_size') == null
          ? const rust.AppVec3(x: 0, y: 0, z: 0)
          : _vec3(_readOptionalObject(json, 'bounds_size')!),
      minResolution: _readInt(json, 'min_resolution', defaultValue: 16),
      maxResolution: _readInt(json, 'max_resolution', defaultValue: 320),
    );

rust.AppImportStatusSnapshot _importStatus(Map<String, dynamic> json) =>
    rust.AppImportStatusSnapshot(
      state: _readString(json, 'state', defaultValue: 'idle'),
      progress: _readInt(json, 'progress'),
      total: _readInt(json, 'total'),
      filename: _readNullableString(json, 'filename'),
      phaseLabel: _readNullableString(json, 'phase_label'),
      message: _readNullableString(json, 'message'),
      isError: _readBool(json, 'is_error'),
    );

rust.AppImportSnapshot _importSnapshot(Map<String, dynamic> json) =>
    rust.AppImportSnapshot(
      dialog: _readOptionalObject(json, 'dialog') == null
          ? null
          : _importDialog(_readOptionalObject(json, 'dialog')!),
      status: _readOptionalObject(json, 'status') == null
          ? _defaultImportSnapshot().status
          : _importStatus(_readOptionalObject(json, 'status')!),
    );

rust.AppSculptConvertDialogSnapshot _sculptConvertDialog(
  Map<String, dynamic> json,
) => rust.AppSculptConvertDialogSnapshot(
  targetNodeId: _readBigInt(json, 'target_node_id'),
  targetName: _readString(json, 'target_name', defaultValue: 'Node'),
  modeId: _readString(json, 'mode_id', defaultValue: 'active_node'),
  modeLabel: _readString(
    json,
    'mode_label',
    defaultValue: 'Bake active node',
  ),
  resolution: _readInt(json, 'resolution', defaultValue: 64),
  minResolution: _readInt(json, 'min_resolution', defaultValue: 16),
  maxResolution: _readInt(json, 'max_resolution', defaultValue: 320),
);

rust.AppSculptConvertStatusSnapshot _sculptConvertStatus(
  Map<String, dynamic> json,
) => rust.AppSculptConvertStatusSnapshot(
  state: _readString(json, 'state', defaultValue: 'idle'),
  progress: _readInt(json, 'progress'),
  total: _readInt(json, 'total'),
  targetName: _readNullableString(json, 'target_name'),
  phaseLabel: _readNullableString(json, 'phase_label'),
  message: _readNullableString(json, 'message'),
  isError: _readBool(json, 'is_error'),
);

rust.AppSculptConvertSnapshot _sculptConvertSnapshot(Map<String, dynamic> json) =>
    rust.AppSculptConvertSnapshot(
      dialog: _readOptionalObject(json, 'dialog') == null
          ? null
          : _sculptConvertDialog(_readOptionalObject(json, 'dialog')!),
      status: _readOptionalObject(json, 'status') == null
          ? _defaultSculptConvertSnapshot().status
          : _sculptConvertStatus(_readOptionalObject(json, 'status')!),
    );

rust.AppSelectedSculptSnapshot _selectedSculpt(Map<String, dynamic> json) =>
    rust.AppSelectedSculptSnapshot(
      nodeId: _readBigInt(json, 'node_id'),
      nodeName: _readString(json, 'node_name'),
      currentResolution: _readInt(json, 'current_resolution'),
      desiredResolution: _readInt(json, 'desired_resolution'),
    );

rust.AppSculptSessionSnapshot _sculptSession(Map<String, dynamic> json) =>
    rust.AppSculptSessionSnapshot(
      nodeId: _readBigInt(json, 'node_id'),
      nodeName: _readString(json, 'node_name'),
      brushModeId: _readString(json, 'brush_mode_id'),
      brushModeLabel: _readString(json, 'brush_mode_label'),
      brushRadius: _readDouble(json, 'brush_radius'),
      brushStrength: _readDouble(json, 'brush_strength'),
      symmetryAxisId: _readString(json, 'symmetry_axis_id'),
      symmetryAxisLabel: _readString(json, 'symmetry_axis_label'),
    );

rust.AppSculptSnapshot _sculptSnapshot(Map<String, dynamic> json) =>
    rust.AppSculptSnapshot(
      selected: _readOptionalObject(json, 'selected') == null
          ? null
          : _selectedSculpt(_readOptionalObject(json, 'selected')!),
      session: _readOptionalObject(json, 'session') == null
          ? null
          : _sculptSession(_readOptionalObject(json, 'session')!),
      canResumeSelected: _readBool(json, 'can_resume_selected'),
      canStop: _readBool(json, 'can_stop'),
      maxResolution: _readInt(json, 'max_resolution', defaultValue: 16),
    );

rust.AppLightCookieCandidateSnapshot _lightCookieCandidate(
  Map<String, dynamic> json,
) => rust.AppLightCookieCandidateSnapshot(
  nodeId: _readBigInt(json, 'node_id'),
  name: _readString(json, 'name'),
  kindLabel: _readString(json, 'kind_label'),
);

rust.AppLightPropertiesSnapshot _lightProperties(Map<String, dynamic> json) =>
    rust.AppLightPropertiesSnapshot(
      nodeId: _readBigInt(json, 'node_id'),
      transformNodeId: json['transform_node_id'] == null
          ? null
          : _readBigInt(json, 'transform_node_id'),
      lightTypeId: _readString(json, 'light_type_id'),
      lightTypeLabel: _readString(json, 'light_type_label'),
      color: _vec3(_readJsonObject(json['color'])),
      intensity: _readDouble(json, 'intensity'),
      range: _readDouble(json, 'range'),
      spotAngle: _readDouble(json, 'spot_angle'),
      castShadows: _readBool(json, 'cast_shadows'),
      shadowSoftness: _readDouble(json, 'shadow_softness'),
      shadowColor: _vec3(_readJsonObject(json['shadow_color'])),
      volumetric: _readBool(json, 'volumetric'),
      volumetricDensity: _readDouble(json, 'volumetric_density'),
      cookieNodeId: json['cookie_node_id'] == null
          ? null
          : _readBigInt(json, 'cookie_node_id'),
      cookieNodeName: _readNullableString(json, 'cookie_node_name'),
      cookieCandidates: _readObjectList(
        json,
        'cookie_candidates',
        _lightCookieCandidate,
      ),
      proximityModeId: _readString(json, 'proximity_mode_id'),
      proximityModeLabel: _readString(json, 'proximity_mode_label'),
      proximityRange: _readDouble(json, 'proximity_range'),
      arrayPatternId: _readNullableString(json, 'array_pattern_id'),
      arrayPatternLabel: _readNullableString(json, 'array_pattern_label'),
      arrayCount: json['array_count'] is num ? _readInt(json, 'array_count') : null,
      arrayRadius: json['array_radius'] is num
          ? _readDouble(json, 'array_radius')
          : null,
      arrayColorVariation: json['array_color_variation'] is num
          ? _readDouble(json, 'array_color_variation')
          : null,
      intensityExpression: _readNullableString(json, 'intensity_expression'),
      intensityExpressionError: _readNullableString(
        json,
        'intensity_expression_error',
      ),
      colorHueExpression: _readNullableString(json, 'color_hue_expression'),
      colorHueExpressionError: _readNullableString(
        json,
        'color_hue_expression_error',
      ),
      supportsRange: _readBool(json, 'supports_range'),
      supportsSpotAngle: _readBool(json, 'supports_spot_angle'),
      supportsShadows: _readBool(json, 'supports_shadows'),
      supportsVolumetric: _readBool(json, 'supports_volumetric'),
      supportsCookie: _readBool(json, 'supports_cookie'),
      supportsProximity: _readBool(json, 'supports_proximity'),
      supportsExpressions: _readBool(json, 'supports_expressions'),
      supportsArray: _readBool(json, 'supports_array'),
    );

rust.AppLightLinkTargetSnapshot _lightLinkTarget(Map<String, dynamic> json) =>
    rust.AppLightLinkTargetSnapshot(
      lightNodeId: _readBigInt(json, 'light_node_id'),
      lightName: _readString(json, 'light_name'),
      lightTypeLabel: _readString(json, 'light_type_label'),
      active: _readBool(json, 'active'),
      maskBit: _readInt(json, 'mask_bit'),
      color: _vec3(_readJsonObject(json['color'])),
    );

rust.AppLightLinkNodeSnapshot _lightLinkNode(Map<String, dynamic> json) =>
    rust.AppLightLinkNodeSnapshot(
      nodeId: _readBigInt(json, 'node_id'),
      nodeName: _readString(json, 'node_name'),
      kindLabel: _readString(json, 'kind_label'),
      lightMask: _readInt(json, 'light_mask'),
    );

rust.AppLightLinkingSnapshot _lightLinking(Map<String, dynamic> json) =>
    rust.AppLightLinkingSnapshot(
      lights: _readObjectList(json, 'lights', _lightLinkTarget),
      geometryNodes: _readObjectList(json, 'geometry_nodes', _lightLinkNode),
      totalVisibleLightCount: _readInt(json, 'total_visible_light_count'),
      maxLightCount: _readInt(json, 'max_light_count', defaultValue: 8),
    );

rust.AppViewportLightSnapshot _viewportLight(Map<String, dynamic> json) =>
    rust.AppViewportLightSnapshot(
      lightNodeId: _readBigInt(json, 'light_node_id'),
      transformNodeId: _readBigInt(json, 'transform_node_id'),
      lightTypeId: _readString(json, 'light_type_id'),
      lightTypeLabel: _readString(json, 'light_type_label'),
      worldPosition: _vec3(_readJsonObject(json['world_position'])),
      direction: _vec3(_readJsonObject(json['direction'])),
      color: _vec3(_readJsonObject(json['color'])),
      intensity: _readDouble(json, 'intensity'),
      range: _readDouble(json, 'range'),
      spotAngle: _readDouble(json, 'spot_angle'),
      active: _readBool(json, 'active', defaultValue: true),
      arrayPositions: _readObjectList(
        json,
        'array_positions',
        _vec3,
      ),
      arrayColors: _readObjectList(
        json,
        'array_colors',
        _vec3,
      ),
    );

rust.AppScalarPropertySnapshot _scalarProperty(Map<String, dynamic> json) =>
    rust.AppScalarPropertySnapshot(
      key: _readString(json, 'key'),
      label: _readString(json, 'label'),
      value: _readDouble(json, 'value'),
    );

rust.AppTransformPropertiesSnapshot _transformProperties(
  Map<String, dynamic> json,
) => rust.AppTransformPropertiesSnapshot(
  positionLabel: _readString(json, 'position_label'),
  position: _vec3(_readJsonObject(json['position'])),
  rotationDegrees: _vec3(_readJsonObject(json['rotation_degrees'])),
  scale: _readOptionalObject(json, 'scale') == null
      ? null
      : _vec3(_readOptionalObject(json, 'scale')!),
);

rust.AppPrimitivePropertiesSnapshot _primitiveProperties(
  Map<String, dynamic> json,
) => rust.AppPrimitivePropertiesSnapshot(
  primitiveKind: _readString(json, 'primitive_kind'),
  parameters: _readObjectList(json, 'parameters', _scalarProperty),
);

rust.AppMaterialPropertiesSnapshot _materialProperties(
  Map<String, dynamic> json,
) => rust.AppMaterialPropertiesSnapshot(
  color: _vec3(_readJsonObject(json['color'])),
  roughness: _readDouble(json, 'roughness'),
  metallic: _readDouble(json, 'metallic'),
  emissive: _vec3(_readJsonObject(json['emissive'])),
  emissiveIntensity: _readDouble(json, 'emissive_intensity'),
  fresnel: _readDouble(json, 'fresnel'),
);

rust.AppSelectedNodePropertiesSnapshot _selectedNodeProperties(
  Map<String, dynamic> json,
) => rust.AppSelectedNodePropertiesSnapshot(
  nodeId: _readBigInt(json, 'node_id'),
  name: _readString(json, 'name'),
  kindLabel: _readString(json, 'kind_label'),
  visible: _readBool(json, 'visible'),
  locked: _readBool(json, 'locked'),
  transform: _readOptionalObject(json, 'transform') == null
      ? null
      : _transformProperties(_readOptionalObject(json, 'transform')!),
  primitive: _readOptionalObject(json, 'primitive') == null
      ? null
      : _primitiveProperties(_readOptionalObject(json, 'primitive')!),
  material: _readOptionalObject(json, 'material') == null
      ? null
      : _materialProperties(_readOptionalObject(json, 'material')!),
  light: _readOptionalObject(json, 'light') == null
      ? null
      : _lightProperties(_readOptionalObject(json, 'light')!),
);

rust.AppSceneSnapshot _sceneSnapshot(Map<String, dynamic> json) {
  final topLevelNodes = _readObjectList(json, 'top_level_nodes', _node);
  final sceneTreeRoots = json['scene_tree_roots'] == null
      ? topLevelNodes.map(_sceneTreeNodeFromNode).toList(growable: false)
      : _readObjectList(json, 'scene_tree_roots', _sceneTreeNode);

  return rust.AppSceneSnapshot(
    selectedNode: _readOptionalObject(json, 'selected_node') == null
        ? null
        : _node(_readOptionalObject(json, 'selected_node')!),
    selectedNodeProperties: _readOptionalObject(json, 'selected_node_properties') ==
            null
        ? null
        : _selectedNodeProperties(
            _readOptionalObject(json, 'selected_node_properties')!,
          ),
    selectedNodeIds: Uint64List.fromList(
      _readJsonList(json['selected_node_ids'])
          .whereType<num>()
          .map((nodeId) => nodeId.toInt())
          .toList(growable: false),
    ),
    topLevelNodes: topLevelNodes,
    sceneTreeRoots: sceneTreeRoots,
    viewportLights: _readObjectList(json, 'viewport_lights', _viewportLight),
    workspace: _readOptionalObject(json, 'workspace') == null
        ? _defaultWorkspaceSnapshot()
        : _workspace(_readOptionalObject(json, 'workspace')!),
    selectionContext: _readOptionalObject(json, 'selection_context') == null
        ? _defaultSelectionContextSnapshot()
        : _selectionContext(_readOptionalObject(json, 'selection_context')!),
    commands: _readObjectList(json, 'commands', _command),
    history: _readOptionalObject(json, 'history') == null
        ? const rust.AppHistorySnapshot(canUndo: false, canRedo: false)
        : _history(_readOptionalObject(json, 'history')!),
    document: _readOptionalObject(json, 'document') == null
        ? _defaultDocumentSnapshot()
        : _document(_readOptionalObject(json, 'document')!),
    render: _readOptionalObject(json, 'render') == null
        ? _defaultRenderSettings()
        : _renderSettings(_readOptionalObject(json, 'render')!),
    settings: _readOptionalObject(json, 'settings') == null
        ? _defaultSettingsSnapshot()
        : _settings(_readOptionalObject(json, 'settings')!),
    export_: _readOptionalObject(json, 'export') == null
        ? _defaultExportSnapshot()
        : _exportSnapshot(_readOptionalObject(json, 'export')!),
    import_: _readOptionalObject(json, 'import') == null
        ? _defaultImportSnapshot()
        : _importSnapshot(_readOptionalObject(json, 'import')!),
    sculptConvert: _readOptionalObject(json, 'sculpt_convert') == null
        ? _defaultSculptConvertSnapshot()
        : _sculptConvertSnapshot(_readOptionalObject(json, 'sculpt_convert')!),
    sculpt: _readOptionalObject(json, 'sculpt') == null
        ? _defaultSculptSnapshot()
        : _sculptSnapshot(_readOptionalObject(json, 'sculpt')!),
    lightLinking: _readOptionalObject(json, 'light_linking') == null
        ? _defaultLightLinkingSnapshot()
        : _lightLinking(_readOptionalObject(json, 'light_linking')!),
    camera: _camera(_readJsonObject(json['camera'])),
    stats: _sceneStats(_readJsonObject(json['stats'])),
    tool: _tool(_readJsonObject(json['tool'])),
  );
}

rust.AppWorkflowStatusSnapshot _workflowStatus(Map<String, dynamic> json) =>
    rust.AppWorkflowStatusSnapshot(
      exportStatus: _readOptionalObject(json, 'export_status') == null
          ? _defaultExportSnapshot().status
          : _exportStatus(_readOptionalObject(json, 'export_status')!),
      importStatus: _readOptionalObject(json, 'import_status') == null
          ? _defaultImportSnapshot().status
          : _importStatus(_readOptionalObject(json, 'import_status')!),
      sculptConvertStatus: _readOptionalObject(json, 'sculpt_convert_status') ==
              null
          ? _defaultSculptConvertSnapshot().status
          : _sculptConvertStatus(
              _readOptionalObject(json, 'sculpt_convert_status')!,
            ),
      sceneChanged: _readBool(json, 'scene_changed'),
    );

rust.AppDocumentSnapshot _defaultDocumentSnapshot() => const rust.AppDocumentSnapshot(
  currentFilePath: null,
  currentFileName: null,
  hasUnsavedChanges: false,
  recentFiles: <String>[],
  recoveryAvailable: false,
  recoverySummary: null,
);

rust.AppRenderSettingsSnapshot _defaultRenderSettings() =>
    const rust.AppRenderSettingsSnapshot(
      shadingModes: <rust.AppRenderOptionSnapshot>[],
      shadingModeId: 'full',
      shadingModeLabel: 'Full',
      showGrid: true,
      shadowsEnabled: false,
      shadowSteps: 32,
      aoEnabled: true,
      aoSamples: 5,
      aoIntensity: 3.0,
      marchMaxSteps: 128,
      sculptFastMode: false,
      autoReduceSteps: true,
      interactionRenderScale: 0.5,
      restRenderScale: 1.0,
      fogEnabled: false,
      fogDensity: 0.02,
      bloomEnabled: false,
      bloomIntensity: 0.3,
      gamma: 2.2,
      tonemappingAces: false,
      crossSectionAxis: 0,
      crossSectionPosition: 0.0,
    );

rust.AppSettingsSnapshot _defaultSettingsSnapshot() => const rust.AppSettingsSnapshot(
  showFpsOverlay: true,
  showNodeLabels: false,
  showBoundingBox: true,
  showLightGizmos: true,
  autoSaveEnabled: true,
  autoSaveIntervalSecs: 120,
  maxExportResolution: 2048,
  maxSculptResolution: 320,
  cameraBookmarks: <rust.AppCameraBookmarkSnapshot>[],
  keyOptions: <rust.AppKeyOptionSnapshot>[],
  keybindings: <rust.AppKeybindingSnapshot>[],
);

rust.AppExportSnapshot _defaultExportSnapshot() => const rust.AppExportSnapshot(
  resolution: 128,
  minResolution: 16,
  maxResolution: 2048,
  adaptive: false,
  presets: <rust.AppExportPresetSnapshot>[],
  status: rust.AppExportStatusSnapshot(
    state: 'idle',
    progress: 0,
    total: 0,
    resolution: 128,
    phaseLabel: null,
    targetFileName: null,
    targetFilePath: null,
    formatLabel: null,
    message: null,
    isError: false,
  ),
);

rust.AppImportSnapshot _defaultImportSnapshot() => const rust.AppImportSnapshot(
  dialog: null,
  status: rust.AppImportStatusSnapshot(
    state: 'idle',
    progress: 0,
    total: 0,
    filename: null,
    phaseLabel: null,
    message: null,
    isError: false,
  ),
);

rust.AppSculptConvertSnapshot _defaultSculptConvertSnapshot() =>
    const rust.AppSculptConvertSnapshot(
      dialog: null,
      status: rust.AppSculptConvertStatusSnapshot(
        state: 'idle',
        progress: 0,
        total: 0,
        targetName: null,
        phaseLabel: null,
        message: null,
        isError: false,
      ),
    );

rust.AppSculptSnapshot _defaultSculptSnapshot() => const rust.AppSculptSnapshot(
  selected: null,
  session: null,
  canResumeSelected: false,
  canStop: false,
  maxResolution: 16,
);

rust.AppWorkspaceSnapshot _defaultWorkspaceSnapshot() =>
    const rust.AppWorkspaceSnapshot(
      id: 'blockout',
      label: 'Blockout',
      description: 'Live SDF primitives, booleans, and transforms.',
    );

rust.AppSelectionContextSnapshot _defaultSelectionContextSnapshot() =>
    const rust.AppSelectionContextSnapshot(
      headline: 'Blockout workspace',
      detail: 'Choose a shape or use the tool rail to start blocking out.',
      selectionCount: 0,
      selectionKindId: 'none',
      selectionKindLabel: 'Nothing selected',
      workflowStatusId: 'none',
      workflowStatusLabel: 'No selection',
      quickActions: <rust.AppQuickActionSnapshot>[],
    );

rust.AppLightLinkingSnapshot _defaultLightLinkingSnapshot() =>
    const rust.AppLightLinkingSnapshot(
      lights: <rust.AppLightLinkTargetSnapshot>[],
      geometryNodes: <rust.AppLightLinkNodeSnapshot>[],
      totalVisibleLightCount: 0,
      maxLightCount: 8,
    );
