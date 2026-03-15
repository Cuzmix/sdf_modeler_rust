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
    expect(snapshot.sculpt.selected, isNull);
    expect(snapshot.sculpt.session, isNull);
    expect(snapshot.sculpt.canResumeSelected, isFalse);
  });

  test('parses sculpt snapshot payloads', () {
    final snapshot = AppSceneSnapshot.fromJson(
      jsonDecode(
            '''
{
  "selected_node": {
    "id": 8,
    "name": "Sculpt",
    "kind_label": "Sculpt",
    "visible": true,
    "locked": false
  },
  "top_level_nodes": [],
  "scene_tree_roots": [],
  "history": {"can_undo": true, "can_redo": false},
  "sculpt": {
    "selected": {
      "node_id": 8,
      "node_name": "Sculpt",
      "current_resolution": 64,
      "desired_resolution": 96
    },
    "session": {
      "node_id": 8,
      "node_name": "Sculpt",
      "brush_mode_id": "grab",
      "brush_mode_label": "Grab",
      "brush_radius": 0.35,
      "brush_strength": 3.0,
      "symmetry_axis_id": "z",
      "symmetry_axis_label": "Z"
    },
    "can_resume_selected": false,
    "can_stop": true,
    "max_resolution": 256
  },
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
    "primitive_nodes": 0,
    "operation_nodes": 0,
    "transform_nodes": 0,
    "modifier_nodes": 0,
    "sculpt_nodes": 1,
    "light_nodes": 0,
    "voxel_memory_bytes": 1048576,
    "sdf_eval_complexity": 1,
    "structure_key": 12,
    "data_fingerprint": 33,
    "bounds_min": {"x": -1.0, "y": -1.0, "z": -1.0},
    "bounds_max": {"x": 1.0, "y": 1.0, "z": 1.0}
  },
  "tool": {
    "active_tool_label": "Sculpt",
    "shading_mode_label": "Full",
    "grid_enabled": true
  }
}
''',
          )
          as Map<String, dynamic>,
    );

    expect(snapshot.sculpt.selected, isNotNull);
    expect(snapshot.sculpt.selected!.desiredResolution, 96);
    expect(snapshot.sculpt.session, isNotNull);
    expect(snapshot.sculpt.session!.brushModeId, 'grab');
    expect(snapshot.sculpt.session!.symmetryAxisId, 'z');
    expect(snapshot.sculpt.canStop, isTrue);
    expect(snapshot.sculpt.maxResolution, 256);
  });

  test('parses advanced light and light-linking snapshots', () {
    final snapshot = AppSceneSnapshot.fromJson(
      jsonDecode(
            '''
{
  "selected_node": {
    "id": 10,
    "name": "Point Light Transform",
    "kind_label": "Transform",
    "visible": true,
    "locked": false
  },
  "selected_node_properties": {
    "node_id": 10,
    "name": "Point Light Transform",
    "kind_label": "Transform",
    "visible": true,
    "locked": false,
    "transform": {
      "position_label": "Translation",
      "position": {"x": 0.0, "y": 2.0, "z": 4.0},
      "rotation_degrees": {"x": 0.0, "y": 0.0, "z": 0.0},
      "scale": {"x": 1.0, "y": 1.0, "z": 1.0}
    },
    "primitive": null,
    "material": null,
    "light": {
      "node_id": 11,
      "transform_node_id": 10,
      "light_type_id": "spot",
      "light_type_label": "Spot",
      "color": {"x": 1.0, "y": 0.8, "z": 0.6},
      "intensity": 2.5,
      "range": 12.0,
      "spot_angle": 35.0,
      "cast_shadows": true,
      "shadow_softness": 16.0,
      "shadow_color": {"x": 0.1, "y": 0.1, "z": 0.2},
      "volumetric": true,
      "volumetric_density": 0.4,
      "cookie_node_id": 3,
      "cookie_node_name": "Cookie Sphere",
      "cookie_candidates": [
        {"node_id": 3, "name": "Cookie Sphere", "kind_label": "Sphere"}
      ],
      "proximity_mode_id": "brighten",
      "proximity_mode_label": "Brighten",
      "proximity_range": 3.0,
      "array_pattern_id": null,
      "array_pattern_label": null,
      "array_count": null,
      "array_radius": null,
      "array_color_variation": null,
      "intensity_expression": "sin(t)",
      "intensity_expression_error": null,
      "color_hue_expression": "fract(t * 0.1) * 360.0",
      "color_hue_expression_error": null,
      "supports_range": true,
      "supports_spot_angle": true,
      "supports_shadows": true,
      "supports_volumetric": true,
      "supports_cookie": true,
      "supports_proximity": true,
      "supports_expressions": true,
      "supports_array": false
    }
  },
  "top_level_nodes": [],
  "scene_tree_roots": [],
  "history": {"can_undo": true, "can_redo": false},
  "light_linking": {
    "lights": [
      {
        "light_node_id": 11,
        "light_name": "Key Light",
        "light_type_label": "Spot",
        "active": true,
        "mask_bit": 1,
        "color": {"x": 1.0, "y": 0.8, "z": 0.6}
      }
    ],
    "geometry_nodes": [
      {
        "node_id": 1,
        "node_name": "Sphere",
        "kind_label": "Sphere",
        "light_mask": 255
      }
    ],
    "total_visible_light_count": 1,
    "max_light_count": 8
  },
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
    "total_nodes": 2,
    "visible_nodes": 2,
    "top_level_nodes": 1,
    "primitive_nodes": 0,
    "operation_nodes": 0,
    "transform_nodes": 1,
    "modifier_nodes": 0,
    "sculpt_nodes": 0,
    "light_nodes": 1,
    "voxel_memory_bytes": 0,
    "sdf_eval_complexity": 1,
    "structure_key": 44,
    "data_fingerprint": 55,
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

    final light = snapshot.selectedNodeProperties!.light;
    expect(light, isNotNull);
    expect(light!.nodeId, 11);
    expect(light.transformNodeId, 10);
    expect(light.lightTypeId, 'spot');
    expect(light.castShadows, isTrue);
    expect(light.cookieCandidates.single.name, 'Cookie Sphere');
    expect(light.supportsCookie, isTrue);
    expect(snapshot.lightLinking.lights.single.maskBit, 1);
    expect(snapshot.lightLinking.geometryNodes.single.lightMask, 255);
    expect(snapshot.lightLinking.maxLightCount, 8);
  });
}
