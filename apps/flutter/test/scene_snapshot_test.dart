import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_snapshot.dart';

void main() {
  test('parses selected node property snapshots', () {
    final snapshot = AppSceneSnapshot.fromJson(
      jsonDecode(
            '''
{
  "selected_node": {
    "id": 1,
    "name": "Hero Sphere",
    "kind_label": "Sphere",
    "visible": true,
    "locked": false
  },
  "selected_node_properties": {
    "node_id": 1,
    "name": "Hero Sphere",
    "kind_label": "Sphere",
    "visible": true,
    "locked": false,
    "transform": {
      "position_label": "Position",
      "position": {"x": 1.0, "y": 2.0, "z": 3.0},
      "rotation_degrees": {"x": 0.0, "y": 45.0, "z": 90.0},
      "scale": null
    },
    "primitive": {
      "primitive_kind": "Sphere",
      "parameters": [
        {"key": "radius", "label": "Radius", "value": 1.5}
      ]
    },
    "material": {
      "color": {"x": 0.8, "y": 0.3, "z": 0.2},
      "roughness": 0.5,
      "metallic": 0.1,
      "emissive": {"x": 0.0, "y": 0.0, "z": 0.0},
      "emissive_intensity": 0.0,
      "fresnel": 0.04
    }
  },
  "top_level_nodes": [
    {
      "id": 1,
      "name": "Hero Sphere",
      "kind_label": "Sphere",
      "visible": true,
      "locked": false
    }
  ],
  "scene_tree_roots": [],
  "history": {"can_undo": true, "can_redo": false},
  "camera": {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "distance": 5.0,
    "fov_degrees": 45.0,
    "orthographic": false,
    "target": {"x": 0.0, "y": 0.0, "z": 0.0},
    "eye": {"x": 3.0, "y": 2.0, "z": 3.0}
  },
  "stats": {
    "total_nodes": 1,
    "visible_nodes": 1,
    "top_level_nodes": 1,
    "primitive_nodes": 1,
    "operation_nodes": 0,
    "transform_nodes": 0,
    "modifier_nodes": 0,
    "sculpt_nodes": 0,
    "light_nodes": 0,
    "voxel_memory_bytes": 0,
    "sdf_eval_complexity": 1,
    "structure_key": 11,
    "data_fingerprint": 22,
    "bounds_min": {"x": -1.0, "y": -1.0, "z": -1.0},
    "bounds_max": {"x": 1.0, "y": 1.0, "z": 1.0}
  },
  "tool": {
    "active_tool_label": "Select",
    "shading_mode_label": "Full",
    "grid_enabled": true
  }
}
''',
          )
          as Map<String, dynamic>,
    );

    final properties = snapshot.selectedNodeProperties;
    expect(properties, isNotNull);
    expect(properties!.nodeId, 1);
    expect(properties.kindLabel, 'Sphere');
    expect(properties.transform!.positionLabel, 'Position');
    expect(properties.transform!.rotationDegrees.y, 45.0);
    expect(properties.primitive!.primitiveKind, 'Sphere');
    expect(properties.primitive!.parameters.single.key, 'radius');
    expect(properties.material!.metallic, 0.1);
  });

  test('keeps selected node properties optional for older snapshots', () {
    final snapshot = AppSceneSnapshot.fromJson(
      jsonDecode(
            '''
{
  "selected_node": null,
  "top_level_nodes": [],
  "scene_tree_roots": [],
  "history": {"can_undo": false, "can_redo": false},
  "camera": {
    "yaw": 0.0,
    "pitch": 0.0,
    "roll": 0.0,
    "distance": 5.0,
    "fov_degrees": 45.0,
    "orthographic": false,
    "target": {"x": 0.0, "y": 0.0, "z": 0.0},
    "eye": {"x": 3.0, "y": 2.0, "z": 3.0}
  },
  "stats": {
    "total_nodes": 0,
    "visible_nodes": 0,
    "top_level_nodes": 0,
    "primitive_nodes": 0,
    "operation_nodes": 0,
    "transform_nodes": 0,
    "modifier_nodes": 0,
    "sculpt_nodes": 0,
    "light_nodes": 0,
    "voxel_memory_bytes": 0,
    "sdf_eval_complexity": 0,
    "structure_key": 0,
    "data_fingerprint": 0,
    "bounds_min": {"x": 0.0, "y": 0.0, "z": 0.0},
    "bounds_max": {"x": 0.0, "y": 0.0, "z": 0.0}
  },
  "tool": {
    "active_tool_label": "Select",
    "shading_mode_label": "Full",
    "grid_enabled": true
  }
}
''',
          )
          as Map<String, dynamic>,
    );

    expect(snapshot.selectedNodeProperties, isNull);
  });
}
