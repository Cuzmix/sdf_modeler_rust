class AppVec3 {
  const AppVec3({required this.x, required this.y, required this.z});

  final double x;
  final double y;
  final double z;

  factory AppVec3.fromJson(Map<String, dynamic> json) {
    return AppVec3(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      z: (json['z'] as num).toDouble(),
    );
  }
}

class AppCameraSnapshot {
  const AppCameraSnapshot({
    required this.yaw,
    required this.pitch,
    required this.roll,
    required this.distance,
    required this.fovDegrees,
    required this.orthographic,
    required this.target,
    required this.eye,
  });

  final double yaw;
  final double pitch;
  final double roll;
  final double distance;
  final double fovDegrees;
  final bool orthographic;
  final AppVec3 target;
  final AppVec3 eye;

  factory AppCameraSnapshot.fromJson(Map<String, dynamic> json) {
    return AppCameraSnapshot(
      yaw: (json['yaw'] as num).toDouble(),
      pitch: (json['pitch'] as num).toDouble(),
      roll: (json['roll'] as num).toDouble(),
      distance: (json['distance'] as num).toDouble(),
      fovDegrees: (json['fov_degrees'] as num).toDouble(),
      orthographic: json['orthographic'] as bool,
      target: AppVec3.fromJson(json['target'] as Map<String, dynamic>),
      eye: AppVec3.fromJson(json['eye'] as Map<String, dynamic>),
    );
  }
}

class AppNodeSnapshot {
  const AppNodeSnapshot({
    required this.id,
    required this.name,
    required this.kindLabel,
    required this.visible,
    required this.locked,
  });

  final int id;
  final String name;
  final String kindLabel;
  final bool visible;
  final bool locked;

  factory AppNodeSnapshot.fromJson(Map<String, dynamic> json) {
    return AppNodeSnapshot(
      id: (json['id'] as num).toInt(),
      name: json['name'] as String,
      kindLabel: json['kind_label'] as String,
      visible: json['visible'] as bool,
      locked: json['locked'] as bool,
    );
  }
}

class AppSceneTreeNodeSnapshot {
  const AppSceneTreeNodeSnapshot({
    required this.id,
    required this.name,
    required this.kindLabel,
    required this.visible,
    required this.locked,
    required this.children,
  });

  final int id;
  final String name;
  final String kindLabel;
  final bool visible;
  final bool locked;
  final List<AppSceneTreeNodeSnapshot> children;

  factory AppSceneTreeNodeSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSceneTreeNodeSnapshot(
      id: (json['id'] as num).toInt(),
      name: json['name'] as String,
      kindLabel: json['kind_label'] as String,
      visible: json['visible'] as bool,
      locked: json['locked'] as bool,
      children: (json['children'] as List<dynamic>? ?? const [])
          .map(
            (item) => AppSceneTreeNodeSnapshot.fromJson(
              item as Map<String, dynamic>,
            ),
          )
          .toList(growable: false),
    );
  }

  factory AppSceneTreeNodeSnapshot.fromNodeSnapshot(AppNodeSnapshot node) {
    return AppSceneTreeNodeSnapshot(
      id: node.id,
      name: node.name,
      kindLabel: node.kindLabel,
      visible: node.visible,
      locked: node.locked,
      children: const [],
    );
  }
}

class AppSceneStatsSnapshot {
  const AppSceneStatsSnapshot({
    required this.totalNodes,
    required this.visibleNodes,
    required this.topLevelNodes,
    required this.primitiveNodes,
    required this.operationNodes,
    required this.transformNodes,
    required this.modifierNodes,
    required this.sculptNodes,
    required this.lightNodes,
    required this.voxelMemoryBytes,
    required this.sdfEvalComplexity,
    required this.structureKey,
    required this.dataFingerprint,
    required this.boundsMin,
    required this.boundsMax,
  });

  final int totalNodes;
  final int visibleNodes;
  final int topLevelNodes;
  final int primitiveNodes;
  final int operationNodes;
  final int transformNodes;
  final int modifierNodes;
  final int sculptNodes;
  final int lightNodes;
  final int voxelMemoryBytes;
  final int sdfEvalComplexity;
  final int structureKey;
  final int dataFingerprint;
  final AppVec3 boundsMin;
  final AppVec3 boundsMax;

  factory AppSceneStatsSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSceneStatsSnapshot(
      totalNodes: (json['total_nodes'] as num).toInt(),
      visibleNodes: (json['visible_nodes'] as num).toInt(),
      topLevelNodes: (json['top_level_nodes'] as num).toInt(),
      primitiveNodes: (json['primitive_nodes'] as num).toInt(),
      operationNodes: (json['operation_nodes'] as num).toInt(),
      transformNodes: (json['transform_nodes'] as num).toInt(),
      modifierNodes: (json['modifier_nodes'] as num).toInt(),
      sculptNodes: (json['sculpt_nodes'] as num).toInt(),
      lightNodes: (json['light_nodes'] as num).toInt(),
      voxelMemoryBytes: (json['voxel_memory_bytes'] as num).toInt(),
      sdfEvalComplexity: (json['sdf_eval_complexity'] as num).toInt(),
      structureKey: (json['structure_key'] as num).toInt(),
      dataFingerprint: (json['data_fingerprint'] as num).toInt(),
      boundsMin: AppVec3.fromJson(json['bounds_min'] as Map<String, dynamic>),
      boundsMax: AppVec3.fromJson(json['bounds_max'] as Map<String, dynamic>),
    );
  }
}

class AppToolSnapshot {
  const AppToolSnapshot({
    required this.activeToolLabel,
    required this.shadingModeLabel,
    required this.gridEnabled,
    this.manipulatorModeId = 'translate',
    this.manipulatorModeLabel = 'Move',
    this.manipulatorSpaceId = 'local',
    this.manipulatorSpaceLabel = 'Local',
    this.manipulatorVisible = false,
    this.canResetPivot = false,
    this.pivotOffset = const AppVec3(x: 0, y: 0, z: 0),
  });

  final String activeToolLabel;
  final String shadingModeLabel;
  final bool gridEnabled;
  final String manipulatorModeId;
  final String manipulatorModeLabel;
  final String manipulatorSpaceId;
  final String manipulatorSpaceLabel;
  final bool manipulatorVisible;
  final bool canResetPivot;
  final AppVec3 pivotOffset;

  factory AppToolSnapshot.fromJson(Map<String, dynamic> json) {
    return AppToolSnapshot(
      activeToolLabel: json['active_tool_label'] as String,
      shadingModeLabel: json['shading_mode_label'] as String,
      gridEnabled: json['grid_enabled'] as bool,
      manipulatorModeId:
          json['manipulator_mode_id'] as String? ?? 'translate',
      manipulatorModeLabel:
          json['manipulator_mode_label'] as String? ?? 'Move',
      manipulatorSpaceId: json['manipulator_space_id'] as String? ?? 'local',
      manipulatorSpaceLabel:
          json['manipulator_space_label'] as String? ?? 'Local',
      manipulatorVisible: json['manipulator_visible'] as bool? ?? false,
      canResetPivot: json['can_reset_pivot'] as bool? ?? false,
      pivotOffset: json['pivot_offset'] == null
          ? const AppVec3(x: 0, y: 0, z: 0)
          : AppVec3.fromJson(json['pivot_offset'] as Map<String, dynamic>),
    );
  }
}

class AppHistorySnapshot {
  const AppHistorySnapshot({required this.canUndo, required this.canRedo});

  final bool canUndo;
  final bool canRedo;

  factory AppHistorySnapshot.fromJson(Map<String, dynamic> json) {
    return AppHistorySnapshot(
      canUndo: json['can_undo'] as bool? ?? false,
      canRedo: json['can_redo'] as bool? ?? false,
    );
  }
}

class AppDocumentSnapshot {
  const AppDocumentSnapshot({
    required this.currentFilePath,
    required this.currentFileName,
    required this.hasUnsavedChanges,
    required this.recentFiles,
    required this.recoveryAvailable,
    required this.recoverySummary,
  });

  final String? currentFilePath;
  final String? currentFileName;
  final bool hasUnsavedChanges;
  final List<String> recentFiles;
  final bool recoveryAvailable;
  final String? recoverySummary;

  factory AppDocumentSnapshot.fromJson(Map<String, dynamic> json) {
    return AppDocumentSnapshot(
      currentFilePath: json['current_file_path'] as String?,
      currentFileName: json['current_file_name'] as String?,
      hasUnsavedChanges: json['has_unsaved_changes'] as bool? ?? false,
      recentFiles: (json['recent_files'] as List<dynamic>? ?? const [])
          .map((item) => item as String)
          .toList(growable: false),
      recoveryAvailable: json['recovery_available'] as bool? ?? false,
      recoverySummary: json['recovery_summary'] as String?,
    );
  }
}

class AppExportPresetSnapshot {
  const AppExportPresetSnapshot({required this.name, required this.resolution});

  final String name;
  final int resolution;

  factory AppExportPresetSnapshot.fromJson(Map<String, dynamic> json) {
    return AppExportPresetSnapshot(
      name: json['name'] as String? ?? 'Preset',
      resolution: (json['resolution'] as num?)?.toInt() ?? 128,
    );
  }
}

class AppExportStatusSnapshot {
  const AppExportStatusSnapshot({
    required this.state,
    required this.progress,
    required this.total,
    required this.resolution,
    required this.phaseLabel,
    required this.targetFileName,
    required this.targetFilePath,
    required this.formatLabel,
    required this.message,
    required this.isError,
  });

  final String state;
  final int progress;
  final int total;
  final int resolution;
  final String? phaseLabel;
  final String? targetFileName;
  final String? targetFilePath;
  final String? formatLabel;
  final String? message;
  final bool isError;

  bool get isInProgress => state == 'in_progress';

  factory AppExportStatusSnapshot.fromJson(Map<String, dynamic> json) {
    return AppExportStatusSnapshot(
      state: json['state'] as String? ?? 'idle',
      progress: (json['progress'] as num?)?.toInt() ?? 0,
      total: (json['total'] as num?)?.toInt() ?? 0,
      resolution: (json['resolution'] as num?)?.toInt() ?? 128,
      phaseLabel: json['phase_label'] as String?,
      targetFileName: json['target_file_name'] as String?,
      targetFilePath: json['target_file_path'] as String?,
      formatLabel: json['format_label'] as String?,
      message: json['message'] as String?,
      isError: json['is_error'] as bool? ?? false,
    );
  }
}

class AppExportSnapshot {
  const AppExportSnapshot({
    required this.resolution,
    required this.minResolution,
    required this.maxResolution,
    required this.adaptive,
    required this.presets,
    required this.status,
  });

  final int resolution;
  final int minResolution;
  final int maxResolution;
  final bool adaptive;
  final List<AppExportPresetSnapshot> presets;
  final AppExportStatusSnapshot status;

  factory AppExportSnapshot.fromJson(Map<String, dynamic> json) {
    return AppExportSnapshot(
      resolution: (json['resolution'] as num?)?.toInt() ?? 128,
      minResolution: (json['min_resolution'] as num?)?.toInt() ?? 16,
      maxResolution: (json['max_resolution'] as num?)?.toInt() ?? 2048,
      adaptive: json['adaptive'] as bool? ?? false,
      presets: (json['presets'] as List<dynamic>? ?? const [])
          .map(
            (item) => AppExportPresetSnapshot.fromJson(
              item as Map<String, dynamic>,
            ),
          )
          .toList(growable: false),
      status: json['status'] == null
          ? const AppExportStatusSnapshot(
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
            )
          : AppExportStatusSnapshot.fromJson(
              json['status'] as Map<String, dynamic>,
            ),
    );
  }
}

class AppImportDialogSnapshot {
  const AppImportDialogSnapshot({
    required this.filename,
    required this.resolution,
    required this.autoResolution,
    required this.useAuto,
    required this.vertexCount,
    required this.triangleCount,
    required this.boundsSize,
    required this.minResolution,
    required this.maxResolution,
  });

  final String filename;
  final int resolution;
  final int autoResolution;
  final bool useAuto;
  final int vertexCount;
  final int triangleCount;
  final AppVec3 boundsSize;
  final int minResolution;
  final int maxResolution;

  factory AppImportDialogSnapshot.fromJson(Map<String, dynamic> json) {
    return AppImportDialogSnapshot(
      filename: json['filename'] as String? ?? 'mesh',
      resolution: (json['resolution'] as num?)?.toInt() ?? 64,
      autoResolution: (json['auto_resolution'] as num?)?.toInt() ?? 64,
      useAuto: json['use_auto'] as bool? ?? true,
      vertexCount: (json['vertex_count'] as num?)?.toInt() ?? 0,
      triangleCount: (json['triangle_count'] as num?)?.toInt() ?? 0,
      boundsSize: json['bounds_size'] == null
          ? const AppVec3(x: 0, y: 0, z: 0)
          : AppVec3.fromJson(json['bounds_size'] as Map<String, dynamic>),
      minResolution: (json['min_resolution'] as num?)?.toInt() ?? 16,
      maxResolution: (json['max_resolution'] as num?)?.toInt() ?? 320,
    );
  }
}

class AppImportStatusSnapshot {
  const AppImportStatusSnapshot({
    required this.state,
    required this.progress,
    required this.total,
    required this.filename,
    required this.phaseLabel,
    required this.message,
    required this.isError,
  });

  final String state;
  final int progress;
  final int total;
  final String? filename;
  final String? phaseLabel;
  final String? message;
  final bool isError;

  bool get isInProgress => state == 'in_progress';

  factory AppImportStatusSnapshot.fromJson(Map<String, dynamic> json) {
    return AppImportStatusSnapshot(
      state: json['state'] as String? ?? 'idle',
      progress: (json['progress'] as num?)?.toInt() ?? 0,
      total: (json['total'] as num?)?.toInt() ?? 0,
      filename: json['filename'] as String?,
      phaseLabel: json['phase_label'] as String?,
      message: json['message'] as String?,
      isError: json['is_error'] as bool? ?? false,
    );
  }
}

class AppImportSnapshot {
  const AppImportSnapshot({required this.dialog, required this.status});

  final AppImportDialogSnapshot? dialog;
  final AppImportStatusSnapshot status;

  factory AppImportSnapshot.fromJson(Map<String, dynamic> json) {
    return AppImportSnapshot(
      dialog: json['dialog'] == null
          ? null
          : AppImportDialogSnapshot.fromJson(
              json['dialog'] as Map<String, dynamic>,
            ),
      status: json['status'] == null
          ? const AppImportStatusSnapshot(
              state: 'idle',
              progress: 0,
              total: 0,
              filename: null,
              phaseLabel: null,
              message: null,
              isError: false,
            )
          : AppImportStatusSnapshot.fromJson(
              json['status'] as Map<String, dynamic>,
            ),
    );
  }
}

class AppSculptConvertDialogSnapshot {
  const AppSculptConvertDialogSnapshot({
    required this.targetNodeId,
    required this.targetName,
    required this.modeId,
    required this.modeLabel,
    required this.resolution,
    required this.minResolution,
    required this.maxResolution,
  });

  final int targetNodeId;
  final String targetName;
  final String modeId;
  final String modeLabel;
  final int resolution;
  final int minResolution;
  final int maxResolution;

  factory AppSculptConvertDialogSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSculptConvertDialogSnapshot(
      targetNodeId: (json['target_node_id'] as num?)?.toInt() ?? 0,
      targetName: json['target_name'] as String? ?? 'Node',
      modeId: json['mode_id'] as String? ?? 'active_node',
      modeLabel: json['mode_label'] as String? ?? 'Bake active node',
      resolution: (json['resolution'] as num?)?.toInt() ?? 64,
      minResolution: (json['min_resolution'] as num?)?.toInt() ?? 16,
      maxResolution: (json['max_resolution'] as num?)?.toInt() ?? 320,
    );
  }
}

class AppSculptConvertStatusSnapshot {
  const AppSculptConvertStatusSnapshot({
    required this.state,
    required this.progress,
    required this.total,
    required this.targetName,
    required this.phaseLabel,
    required this.message,
    required this.isError,
  });

  final String state;
  final int progress;
  final int total;
  final String? targetName;
  final String? phaseLabel;
  final String? message;
  final bool isError;

  bool get isInProgress => state == 'in_progress';

  factory AppSculptConvertStatusSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSculptConvertStatusSnapshot(
      state: json['state'] as String? ?? 'idle',
      progress: (json['progress'] as num?)?.toInt() ?? 0,
      total: (json['total'] as num?)?.toInt() ?? 0,
      targetName: json['target_name'] as String?,
      phaseLabel: json['phase_label'] as String?,
      message: json['message'] as String?,
      isError: json['is_error'] as bool? ?? false,
    );
  }
}

class AppSculptConvertSnapshot {
  const AppSculptConvertSnapshot({required this.dialog, required this.status});

  final AppSculptConvertDialogSnapshot? dialog;
  final AppSculptConvertStatusSnapshot status;

  factory AppSculptConvertSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSculptConvertSnapshot(
      dialog: json['dialog'] == null
          ? null
          : AppSculptConvertDialogSnapshot.fromJson(
              json['dialog'] as Map<String, dynamic>,
            ),
      status: json['status'] == null
          ? const AppSculptConvertStatusSnapshot(
              state: 'idle',
              progress: 0,
              total: 0,
              targetName: null,
              phaseLabel: null,
              message: null,
              isError: false,
            )
          : AppSculptConvertStatusSnapshot.fromJson(
              json['status'] as Map<String, dynamic>,
            ),
    );
  }
}

class AppScalarPropertySnapshot {
  const AppScalarPropertySnapshot({
    required this.key,
    required this.label,
    required this.value,
  });

  final String key;
  final String label;
  final double value;

  factory AppScalarPropertySnapshot.fromJson(Map<String, dynamic> json) {
    return AppScalarPropertySnapshot(
      key: json['key'] as String,
      label: json['label'] as String,
      value: (json['value'] as num).toDouble(),
    );
  }
}

class AppTransformPropertiesSnapshot {
  const AppTransformPropertiesSnapshot({
    required this.positionLabel,
    required this.position,
    required this.rotationDegrees,
    required this.scale,
  });

  final String positionLabel;
  final AppVec3 position;
  final AppVec3 rotationDegrees;
  final AppVec3? scale;

  factory AppTransformPropertiesSnapshot.fromJson(Map<String, dynamic> json) {
    return AppTransformPropertiesSnapshot(
      positionLabel: json['position_label'] as String,
      position: AppVec3.fromJson(json['position'] as Map<String, dynamic>),
      rotationDegrees: AppVec3.fromJson(
        json['rotation_degrees'] as Map<String, dynamic>,
      ),
      scale: json['scale'] == null
          ? null
          : AppVec3.fromJson(json['scale'] as Map<String, dynamic>),
    );
  }
}

class AppPrimitivePropertiesSnapshot {
  const AppPrimitivePropertiesSnapshot({
    required this.primitiveKind,
    required this.parameters,
  });

  final String primitiveKind;
  final List<AppScalarPropertySnapshot> parameters;

  factory AppPrimitivePropertiesSnapshot.fromJson(Map<String, dynamic> json) {
    return AppPrimitivePropertiesSnapshot(
      primitiveKind: json['primitive_kind'] as String,
      parameters: (json['parameters'] as List<dynamic>? ?? const [])
          .map(
            (item) => AppScalarPropertySnapshot.fromJson(
              item as Map<String, dynamic>,
            ),
          )
          .toList(growable: false),
    );
  }
}

class AppMaterialPropertiesSnapshot {
  const AppMaterialPropertiesSnapshot({
    required this.color,
    required this.roughness,
    required this.metallic,
    required this.emissive,
    required this.emissiveIntensity,
    required this.fresnel,
  });

  final AppVec3 color;
  final double roughness;
  final double metallic;
  final AppVec3 emissive;
  final double emissiveIntensity;
  final double fresnel;

  factory AppMaterialPropertiesSnapshot.fromJson(Map<String, dynamic> json) {
    return AppMaterialPropertiesSnapshot(
      color: AppVec3.fromJson(json['color'] as Map<String, dynamic>),
      roughness: (json['roughness'] as num).toDouble(),
      metallic: (json['metallic'] as num).toDouble(),
      emissive: AppVec3.fromJson(json['emissive'] as Map<String, dynamic>),
      emissiveIntensity: (json['emissive_intensity'] as num).toDouble(),
      fresnel: (json['fresnel'] as num).toDouble(),
    );
  }
}

class AppSelectedNodePropertiesSnapshot {
  const AppSelectedNodePropertiesSnapshot({
    required this.nodeId,
    required this.name,
    required this.kindLabel,
    required this.visible,
    required this.locked,
    required this.transform,
    required this.primitive,
    required this.material,
  });

  final int nodeId;
  final String name;
  final String kindLabel;
  final bool visible;
  final bool locked;
  final AppTransformPropertiesSnapshot? transform;
  final AppPrimitivePropertiesSnapshot? primitive;
  final AppMaterialPropertiesSnapshot? material;

  factory AppSelectedNodePropertiesSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSelectedNodePropertiesSnapshot(
      nodeId: (json['node_id'] as num).toInt(),
      name: json['name'] as String,
      kindLabel: json['kind_label'] as String,
      visible: json['visible'] as bool,
      locked: json['locked'] as bool,
      transform: json['transform'] == null
          ? null
          : AppTransformPropertiesSnapshot.fromJson(
              json['transform'] as Map<String, dynamic>,
            ),
      primitive: json['primitive'] == null
          ? null
          : AppPrimitivePropertiesSnapshot.fromJson(
              json['primitive'] as Map<String, dynamic>,
            ),
      material: json['material'] == null
          ? null
          : AppMaterialPropertiesSnapshot.fromJson(
              json['material'] as Map<String, dynamic>,
            ),
    );
  }
}

class AppSceneSnapshot {
  const AppSceneSnapshot({
    required this.selectedNode,
    required this.selectedNodeProperties,
    required this.topLevelNodes,
    required this.sceneTreeRoots,
    required this.history,
    required this.document,
    required this.export,
    required this.import,
    required this.sculptConvert,
    required this.camera,
    required this.stats,
    required this.tool,
  });

  final AppNodeSnapshot? selectedNode;
  final AppSelectedNodePropertiesSnapshot? selectedNodeProperties;
  final List<AppNodeSnapshot> topLevelNodes;
  final List<AppSceneTreeNodeSnapshot> sceneTreeRoots;
  final AppHistorySnapshot history;
  final AppDocumentSnapshot document;
  final AppExportSnapshot export;
  final AppImportSnapshot import;
  final AppSculptConvertSnapshot sculptConvert;
  final AppCameraSnapshot camera;
  final AppSceneStatsSnapshot stats;
  final AppToolSnapshot tool;

  factory AppSceneSnapshot.fromJson(Map<String, dynamic> json) {
    final topLevelNodes = (json['top_level_nodes'] as List<dynamic>)
        .map((item) => AppNodeSnapshot.fromJson(item as Map<String, dynamic>))
        .toList(growable: false);
    final sceneTreeRootsJson = json['scene_tree_roots'] as List<dynamic>?;

    return AppSceneSnapshot(
      selectedNode: json['selected_node'] == null
          ? null
          : AppNodeSnapshot.fromJson(
              json['selected_node'] as Map<String, dynamic>,
            ),
      selectedNodeProperties: json['selected_node_properties'] == null
          ? null
          : AppSelectedNodePropertiesSnapshot.fromJson(
              json['selected_node_properties'] as Map<String, dynamic>,
            ),
      topLevelNodes: topLevelNodes,
      sceneTreeRoots: sceneTreeRootsJson == null
          ? topLevelNodes
              .map(AppSceneTreeNodeSnapshot.fromNodeSnapshot)
              .toList(growable: false)
          : sceneTreeRootsJson
              .map(
                (item) => AppSceneTreeNodeSnapshot.fromJson(
                  item as Map<String, dynamic>,
                ),
              )
              .toList(growable: false),
      history: json['history'] == null
          ? const AppHistorySnapshot(canUndo: false, canRedo: false)
          : AppHistorySnapshot.fromJson(json['history'] as Map<String, dynamic>),
      document: json['document'] == null
          ? const AppDocumentSnapshot(
              currentFilePath: null,
              currentFileName: null,
              hasUnsavedChanges: false,
              recentFiles: <String>[],
              recoveryAvailable: false,
              recoverySummary: null,
            )
          : AppDocumentSnapshot.fromJson(json['document'] as Map<String, dynamic>),
      export: json['export'] == null
          ? const AppExportSnapshot(
              resolution: 128,
              minResolution: 16,
              maxResolution: 2048,
              adaptive: false,
              presets: <AppExportPresetSnapshot>[],
              status: AppExportStatusSnapshot(
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
            )
          : AppExportSnapshot.fromJson(json['export'] as Map<String, dynamic>),
      import: json['import'] == null
          ? const AppImportSnapshot(
              dialog: null,
              status: AppImportStatusSnapshot(
                state: 'idle',
                progress: 0,
                total: 0,
                filename: null,
                phaseLabel: null,
                message: null,
                isError: false,
              ),
            )
          : AppImportSnapshot.fromJson(json['import'] as Map<String, dynamic>),
      sculptConvert: json['sculpt_convert'] == null
          ? const AppSculptConvertSnapshot(
              dialog: null,
              status: AppSculptConvertStatusSnapshot(
                state: 'idle',
                progress: 0,
                total: 0,
                targetName: null,
                phaseLabel: null,
                message: null,
                isError: false,
              ),
            )
          : AppSculptConvertSnapshot.fromJson(
              json['sculpt_convert'] as Map<String, dynamic>,
            ),
      camera: AppCameraSnapshot.fromJson(json['camera'] as Map<String, dynamic>),
      stats: AppSceneStatsSnapshot.fromJson(
        json['stats'] as Map<String, dynamic>,
      ),
      tool: AppToolSnapshot.fromJson(json['tool'] as Map<String, dynamic>),
    );
  }
}
