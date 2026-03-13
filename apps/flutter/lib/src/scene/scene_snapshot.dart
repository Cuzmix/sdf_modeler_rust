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
  });

  final String activeToolLabel;
  final String shadingModeLabel;
  final bool gridEnabled;

  factory AppToolSnapshot.fromJson(Map<String, dynamic> json) {
    return AppToolSnapshot(
      activeToolLabel: json['active_tool_label'] as String,
      shadingModeLabel: json['shading_mode_label'] as String,
      gridEnabled: json['grid_enabled'] as bool,
    );
  }
}

class AppSceneSnapshot {
  const AppSceneSnapshot({
    required this.selectedNode,
    required this.topLevelNodes,
    required this.camera,
    required this.stats,
    required this.tool,
  });

  final AppNodeSnapshot? selectedNode;
  final List<AppNodeSnapshot> topLevelNodes;
  final AppCameraSnapshot camera;
  final AppSceneStatsSnapshot stats;
  final AppToolSnapshot tool;

  factory AppSceneSnapshot.fromJson(Map<String, dynamic> json) {
    return AppSceneSnapshot(
      selectedNode: json['selected_node'] == null
          ? null
          : AppNodeSnapshot.fromJson(
              json['selected_node'] as Map<String, dynamic>,
            ),
      topLevelNodes: (json['top_level_nodes'] as List<dynamic>)
          .map((item) => AppNodeSnapshot.fromJson(item as Map<String, dynamic>))
          .toList(growable: false),
      camera: AppCameraSnapshot.fromJson(json['camera'] as Map<String, dynamic>),
      stats: AppSceneStatsSnapshot.fromJson(
        json['stats'] as Map<String, dynamic>,
      ),
      tool: AppToolSnapshot.fromJson(json['tool'] as Map<String, dynamic>),
    );
  }
}

