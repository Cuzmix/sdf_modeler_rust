import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/app.dart';
import 'package:sdf_modeler_flutter/src/rust/api/simple.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_desktop_side_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_modal_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_stacked_panes.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

class _MockRustApi extends RustLibApi {
  static const String _baseSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertySnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":1.0}]},"material":{"color":{"x":0.8,"y":0.3,"z":0.2},"roughness":0.5,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertyHiddenSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":false,"locked":false},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":false,"locked":false,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":1.0}]},"material":{"color":{"x":0.8,"y":0.3,"z":0.2},"roughness":0.5,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":false,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":6,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":23,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertyLockedSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":true},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":true,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":1.0}]},"material":{"color":{"x":0.8,"y":0.3,"z":0.2},"roughness":0.5,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":true}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertyRadiusSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":2.5}]},"material":{"color":{"x":0.8,"y":0.3,"z":0.2},"roughness":0.5,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":25,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertyRoughnessSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":1.0}]},"material":{"color":{"x":0.8,"y":0.3,"z":0.2},"roughness":0.8,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":26,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedPropertyColorSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"selected_node_properties":{"node_id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"transform":{"position_label":"Position","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":null},"primitive":{"primitive_kind":"Sphere","parameters":[{"key":"radius","label":"Radius","value":1.0}]},"material":{"color":{"x":0.95,"y":0.3,"z":0.2},"roughness":0.5,"metallic":0.0,"emissive":{"x":0.0,"y":0.0,"z":0.0},"emissive_intensity":0.0,"fresnel":0.04}},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":27,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _baseUndoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedUndoSnapshot = '''{"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _renamedSnapshot = '''{"selected_node":{"id":1,"name":"Hero Sphere","kind_label":"Sphere","visible":true,"locked":false},"top_level_nodes":[{"id":1,"name":"Hero Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":23,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _duplicatedSnapshot = '''{"selected_node":{"id":2,"name":"Sphere Copy","kind_label":"Sphere","visible":true,"locked":false},"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},{"id":2,"name":"Sphere Copy","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":2,"primitive_nodes":2,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":23,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":3.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _operationSnapshot = '''{"selected_node":{"id":8,"name":"Union","kind_label":"Union","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Union","kind_label":"Union","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Union","kind_label":"Union","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]},{"id":2,"name":"Box","kind_label":"Box","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":9,"visible_nodes":9,"top_level_nodes":4,"primitive_nodes":2,"operation_nodes":1,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":3,"structure_key":13,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":3.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _transformSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformPropertySnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformMovedSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":28,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformRotatedSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":10.0,"y":20.0,"z":30.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":29,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformScaledSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":10.0,"y":20.0,"z":30.0},"scale":{"x":0.01,"y":2.0,"z":100.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":30,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _modifierSnapshot = '''{"selected_node":{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":1,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _lightSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":9,"name":"Point","kind_label":"Point","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":9,"visible_nodes":9,"top_level_nodes":5,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":4,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":3.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _sculptSnapshot = '''{"selected_node":{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":1,"light_nodes":3,"voxel_memory_bytes":1048576,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _baseRedoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":true},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _orthoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":true,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';

  String currentSnapshot = _baseSnapshot;
  int addBoxCalls = 0;
  int createOperationCalls = 0;
  int createTransformCalls = 0;
  int createModifierCalls = 0;
  int createLightCalls = 0;
  int createSculptCalls = 0;
  int duplicateSelectedCalls = 0;
  int renameNodeCalls = 0;
  int setSelectedPrimitiveParameterCalls = 0;
  int setSelectedMaterialFloatCalls = 0;
  int setSelectedMaterialColorCalls = 0;
  int setSelectedTransformPositionCalls = 0;
  int setSelectedTransformRotationDegreesCalls = 0;
  int setSelectedTransformScaleCalls = 0;
  int deleteSelectedCalls = 0;
  int focusSelectedCalls = 0;
  int cameraFrontCalls = 0;
  int selectNodeCalls = 0;
  int toggleNodeVisibilityCalls = 0;
  int toggleNodeLockCalls = 0;
  int toggleOrthographicCalls = 0;
  int undoCalls = 0;
  int redoCalls = 0;

  void resetState() {
    currentSnapshot = _baseSnapshot;
    addBoxCalls = 0;
    createOperationCalls = 0;
    createTransformCalls = 0;
    createModifierCalls = 0;
    createLightCalls = 0;
    createSculptCalls = 0;
    duplicateSelectedCalls = 0;
    renameNodeCalls = 0;
    setSelectedPrimitiveParameterCalls = 0;
    setSelectedMaterialFloatCalls = 0;
    setSelectedMaterialColorCalls = 0;
    setSelectedTransformPositionCalls = 0;
    setSelectedTransformRotationDegreesCalls = 0;
    setSelectedTransformScaleCalls = 0;
    deleteSelectedCalls = 0;
    focusSelectedCalls = 0;
    cameraFrontCalls = 0;
    selectNodeCalls = 0;
    toggleNodeVisibilityCalls = 0;
    toggleNodeLockCalls = 0;
    toggleOrthographicCalls = 0;
    undoCalls = 0;
    redoCalls = 0;
  }

  @override
  String crateApiSimpleAddBox() {
    addBoxCalls += 1;
    currentSnapshot = _selectedUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleAddCylinder() => currentSnapshot = _selectedUndoSnapshot;

  @override
  String crateApiSimpleAddSphere() {
    currentSnapshot = _selectedUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleAddTorus() => currentSnapshot = _selectedUndoSnapshot;

  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  String crateApiSimpleCameraBack() => currentSnapshot;

  @override
  String crateApiSimpleCameraBottom() => currentSnapshot;

  @override
  String crateApiSimpleCameraFront() {
    cameraFrontCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCameraLeft() => currentSnapshot;

  @override
  String crateApiSimpleCameraRight() => currentSnapshot;

  @override
  String crateApiSimpleCameraTop() => currentSnapshot;

  @override
  String crateApiSimpleCreateModifier({required String modifierId}) {
    createModifierCalls += 1;
    currentSnapshot = _modifierSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCreateLight({required String lightId}) {
    createLightCalls += 1;
    currentSnapshot = _lightSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCreateOperation({required String operationId}) {
    createOperationCalls += 1;
    currentSnapshot = _operationSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCreateTransform() {
    createTransformCalls += 1;
    currentSnapshot = _transformSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCreateSculpt() {
    createSculptCalls += 1;
    currentSnapshot = _sculptSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleDeleteSelected() {
    deleteSelectedCalls += 1;
    currentSnapshot = _baseUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleDuplicateSelected() {
    duplicateSelectedCalls += 1;
    currentSnapshot = _duplicatedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleRenameNode({
    required BigInt nodeId,
    required String name,
  }) {
    renameNodeCalls += 1;
    currentSnapshot = _renamedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedPrimitiveParameter({
    required String parameterKey,
    required double value,
  }) {
    setSelectedPrimitiveParameterCalls += 1;
    currentSnapshot = _selectedPropertyRadiusSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedMaterialFloat({
    required String fieldId,
    required double value,
  }) {
    setSelectedMaterialFloatCalls += 1;
    currentSnapshot = _selectedPropertyRoughnessSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedMaterialColor({
    required String fieldId,
    required double red,
    required double green,
    required double blue,
  }) {
    setSelectedMaterialColorCalls += 1;
    currentSnapshot = _selectedPropertyColorSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedTransformPosition({
    required double x,
    required double y,
    required double z,
  }) {
    setSelectedTransformPositionCalls += 1;
    currentSnapshot = _selectedTransformMovedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedTransformRotationDegrees({
    required double xDegrees,
    required double yDegrees,
    required double zDegrees,
  }) {
    setSelectedTransformRotationDegreesCalls += 1;
    currentSnapshot = _selectedTransformRotatedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedTransformScale({
    required double x,
    required double y,
    required double z,
  }) {
    setSelectedTransformScaleCalls += 1;
    currentSnapshot = _selectedTransformScaledSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleFocusSelected() {
    focusSelectedCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleFrameAll() => currentSnapshot;

  @override
  Future<void> crateApiSimpleInitApp() async {}

  @override
  void crateApiSimpleOrbitCamera({
    required double deltaX,
    required double deltaY,
  }) {}

  @override
  void crateApiSimplePanCamera({
    required double deltaX,
    required double deltaY,
  }) {}

  @override
  String crateApiSimplePing() => 'pong-test';

  @override
  Future<Uint8List> crateApiSimpleRenderPreviewFrame({
    required int width,
    required int height,
    required double timeSeconds,
  }) async {
    return Uint8List(width * height * 4);
  }

  @override
  String crateApiSimpleResetScene() {
    currentSnapshot = _baseSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleRedo() {
    redoCalls += 1;
    currentSnapshot = _selectedUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSceneSnapshotJson() => currentSnapshot;

  @override
  String crateApiSimpleSelectNode({BigInt? nodeId}) {
    selectNodeCalls += 1;
    currentSnapshot = nodeId == null ? _baseSnapshot : _selectedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSelectNodeAtViewport({
    required double mouseX,
    required double mouseY,
    required int width,
    required int height,
    required double timeSeconds,
  }) {
    currentSnapshot = _selectedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleToggleNodeLock({required BigInt nodeId}) {
    toggleNodeLockCalls += 1;
    currentSnapshot = switch (currentSnapshot) {
      _selectedPropertySnapshot => _selectedPropertyLockedSnapshot,
      _selectedPropertyLockedSnapshot => _selectedPropertySnapshot,
      _ => _selectedUndoSnapshot,
    };
    return currentSnapshot;
  }

  @override
  String crateApiSimpleToggleNodeVisibility({required BigInt nodeId}) {
    toggleNodeVisibilityCalls += 1;
    currentSnapshot = switch (currentSnapshot) {
      _selectedPropertySnapshot => _selectedPropertyHiddenSnapshot,
      _selectedPropertyHiddenSnapshot => _selectedPropertySnapshot,
      _ => _selectedUndoSnapshot,
    };
    return currentSnapshot;
  }

  @override
  String crateApiSimpleToggleOrthographic() {
    toggleOrthographicCalls += 1;
    currentSnapshot = currentSnapshot == _orthoSnapshot
        ? _baseSnapshot
        : _orthoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleUndo() {
    undoCalls += 1;
    currentSnapshot = _baseRedoSnapshot;
    return currentSnapshot;
  }

  @override
  void crateApiSimpleZoomCamera({required double delta}) {}
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  const textureChannel = MethodChannel('sdf_modeler/texture');
  const textureEventChannel = EventChannel('sdf_modeler/texture_events');
  final mockApi = _MockRustApi();

  var createTextureCalls = 0;
  var setTextureSizeCalls = 0;
  var requestFrameCalls = 0;
  var orbitCalls = 0;
  var panCalls = 0;
  var zoomCalls = 0;
  var pickCalls = 0;
  var hoverCalls = 0;
  var clearHoverCalls = 0;
  var disposeTextureCalls = 0;
  MockStreamHandlerEventSink? textureEventSink;

  void resetTextureChannelCounters() {
    createTextureCalls = 0;
    setTextureSizeCalls = 0;
    requestFrameCalls = 0;
    orbitCalls = 0;
    panCalls = 0;
    zoomCalls = 0;
    pickCalls = 0;
    hoverCalls = 0;
    clearHoverCalls = 0;
    disposeTextureCalls = 0;
  }

  Future<void> pumpApp(
    WidgetTester tester, {
    Size logicalSize = const Size(900, 900),
  }) async {
    tester.view.devicePixelRatio = 1.0;
    tester.view.physicalSize = logicalSize;
    addTearDown(() {
      tester.view.resetPhysicalSize();
      tester.view.resetDevicePixelRatio();
    });

    await tester.pumpWidget(const SdfModelerApp());
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 200));
  }

  Future<void> revealTabletBottomSheetContent(
    WidgetTester tester,
    Finder finder,
  ) async {
    await tester.scrollUntilVisible(
      finder,
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();
  }

  setUpAll(() {
    RustLib.initMock(api: mockApi);

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, (call) async {
          switch (call.method) {
            case 'createTexture':
              createTextureCalls += 1;
              return 7;
            case 'setTextureSize':
              setTextureSizeCalls += 1;
              return null;
            case 'requestFrame':
              requestFrameCalls += 1;
              return null;
            case 'orbitCamera':
              orbitCalls += 1;
              return null;
            case 'panCamera':
              panCalls += 1;
              return null;
            case 'zoomCamera':
              zoomCalls += 1;
              return null;
            case 'pickNode':
              pickCalls += 1;
              return null;
            case 'hoverNode':
              hoverCalls += 1;
              return null;
            case 'clearHover':
              clearHoverCalls += 1;
              return null;
            case 'disposeTexture':
              disposeTextureCalls += 1;
              return null;
            case 'updateTexture':
              return null;
            default:
              throw MissingPluginException('Unhandled method: ${call.method}');
          }
        });

    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockStreamHandler(
          textureEventChannel,
          MockStreamHandler.inline(
            onListen: (arguments, events) {
              textureEventSink = events;
            },
            onCancel: (arguments) {
              textureEventSink = null;
            },
          ),
        );
  });

  setUp(() {
    resetTextureChannelCounters();
    textureEventSink = null;
    mockApi.resetState();
  });

  tearDownAll(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(textureChannel, null);
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockStreamHandler(textureEventChannel, null);
  });

  test('routes direct transform commands through the generated Rust facade', () {
    mockApi.currentSnapshot = _MockRustApi._selectedTransformPropertySnapshot;

    final movedSnapshot = setSelectedTransformPosition(x: 1.0, y: -2.0, z: 3.5);
    final rotatedSnapshot = setSelectedTransformRotationDegrees(
      xDegrees: 10.0,
      yDegrees: 20.0,
      zDegrees: 30.0,
    );
    final scaledSnapshot = setSelectedTransformScale(x: -1.0, y: 2.0, z: 500.0);

    expect(mockApi.setSelectedTransformPositionCalls, 1);
    expect(mockApi.setSelectedTransformRotationDegreesCalls, 1);
    expect(mockApi.setSelectedTransformScaleCalls, 1);
    expect(movedSnapshot, _MockRustApi._selectedTransformMovedSnapshot);
    expect(rotatedSnapshot, _MockRustApi._selectedTransformRotatedSnapshot);
    expect(scaledSnapshot, _MockRustApi._selectedTransformScaledSnapshot);
  });

  testWidgets('routes transform inspector edits through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedTransformPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-position-x-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-position-x-slider')),
      const Offset(180, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedTransformPositionCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformMovedSnapshot);

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-rotation-z-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-rotation-z-slider')),
      const Offset(120, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedTransformRotationDegreesCalls, 1);
    expect(
      mockApi.currentSnapshot,
      _MockRustApi._selectedTransformRotatedSnapshot,
    );

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-scale-x-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-scale-x-slider')),
      const Offset(-200, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedTransformScaleCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformScaledSnapshot);
  });

  testWidgets('renders bridge status, scene snapshot, and real viewport host', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    expect(find.textContaining('Rust ping: pong-test'), findsOneWidget);
    expect(
      find.textContaining('Bridge crate version: 0.1.0-test'),
      findsOneWidget,
    );
    expect(find.text('Adaptive Interaction Resolution'), findsOneWidget);
    await revealTabletBottomSheetContent(tester, find.text('Scene Tree'));
    expect(find.text('Scene Tree'), findsOneWidget);
    expect(createTextureCalls, 1);
    expect(setTextureSizeCalls, greaterThanOrEqualTo(0));
    expect(requestFrameCalls, greaterThan(0));
    expect(textureEventSink, isNotNull);

    await tester.pumpWidget(const SizedBox.shrink());
    await tester.pump();
    expect(disposeTextureCalls, 1);
  });

  testWidgets('shows FPS stats in the viewport overlay', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(
      const Directionality(
        textDirection: TextDirection.ltr,
        child: ViewportFeedbackOverlay(
          feedback: null,
          interactionPhase: 'interacting',
          frameTimeMs: 16.0,
          framesPerSecond: 62.5,
          droppedFrameCount: 0,
        ),
      ),
    );

    expect(find.textContaining('62.5 FPS', findRichText: true), findsOneWidget);
    expect(find.textContaining('16.0 ms', findRichText: true), findsOneWidget);
  });

  testWidgets('adaptive interaction resolution only drops scale when enabled', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    await tester.tap(find.text('Adaptive Interaction Resolution'));
    await tester.pump();

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));
    final orbitGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await orbitGesture.moveBy(const Offset(24, 12));
    await tester.pump();

    expect(find.textContaining('at 65% scale interactive'), findsOneWidget);

    await orbitGesture.up();
  });

  testWidgets('shared shell theme keeps command controls touch-sized', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);
    await revealTabletBottomSheetContent(
      tester,
      find.byKey(const ValueKey('open-command-panel')),
    );

    final commandButtonSize = tester.getSize(
      find.byKey(const ValueKey('open-command-panel')),
    );

    expect(
      commandButtonSize.height,
      greaterThanOrEqualTo(ShellTokens.minimumTouchTarget),
    );
  });

  testWidgets('routes scene tree commands through the Rust facade', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('scene-tree-node-1')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();

    await tester.tap(find.byKey(const ValueKey('scene-tree-node-1')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('scene-tree-visibility-1')));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.text('Box'),
      -200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.text('Box'));
    await tester.pump();

    expect(mockApi.selectNodeCalls, 1);
    expect(mockApi.toggleNodeVisibilityCalls, 1);
    expect(mockApi.addBoxCalls, 1);
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('renders backend-owned node basics and routes visibility toggle', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('node-basics-visible-toggle')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('Radius: 1.00'),
      -120,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    expect(find.text('Radius: 1.00'), findsOneWidget);

    await tester.ensureVisible(
      find.byKey(const ValueKey('node-basics-visible-toggle')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('node-basics-visible-toggle')));
    await tester.pumpAndSettle();

    expect(mockApi.toggleNodeVisibilityCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyHiddenSnapshot);
  });

  testWidgets('routes node basics lock toggle through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('node-basics-lock-toggle')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('node-basics-lock-toggle')));
    await tester.pumpAndSettle();

    expect(mockApi.toggleNodeLockCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyLockedSnapshot);
  });

  testWidgets('routes primitive parameter edits through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('primitive-parameter-radius-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('primitive-parameter-radius-slider')),
      const Offset(220, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedPrimitiveParameterCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyRadiusSnapshot);
  });

  testWidgets('routes material scalar edits through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('material-roughness-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('material-roughness-slider')),
      const Offset(180, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedMaterialFloatCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyRoughnessSnapshot);
  });

  testWidgets('routes material color edits through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('material-color-red-slider')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('material-color-red-slider')),
      const Offset(120, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.setSelectedMaterialColorCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyColorSnapshot);
  });

  testWidgets('routes undo through the Rust facade and refreshes snapshot state', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.tap(find.text('Box'));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('undo-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('undo-command')));
    await tester.pump();

    expect(mockApi.addBoxCalls, 1);
    expect(mockApi.undoCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._baseRedoSnapshot);
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets(
    'routes rename through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('rename-command')),
        200,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.pump();
      await tester.tap(find.byKey(const ValueKey('rename-command')));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const ValueKey('rename-node-field')),
        'Hero Sphere',
      );
      await tester.tap(find.byKey(const ValueKey('rename-node-submit')));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 200));
      await tester.scrollUntilVisible(
        find.text('Selected: Hero Sphere'),
        200,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.pumpAndSettle();

      expect(mockApi.renameNodeCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._renamedSnapshot);
      expect(find.text('Selected: Hero Sphere'), findsOneWidget);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes duplicate through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('duplicate-command')),
        200,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.pump();
      await tester.tap(find.byKey(const ValueKey('duplicate-command')));
      await tester.pump();

      expect(mockApi.duplicateSelectedCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._duplicatedSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes create transform through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.tap(find.byKey(const ValueKey('create-transform-command')));
      await tester.pump();

      expect(mockApi.createTransformCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._transformSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes create operation through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.tap(find.byKey(const ValueKey('create-operation-command')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('operation-option-union')));
      await tester.pump();

      expect(mockApi.createOperationCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._operationSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes create modifier through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.tap(find.byKey(const ValueKey('create-modifier-command')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('modifier-option-twist')));
      await tester.pump();

      expect(mockApi.createModifierCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._modifierSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes create light through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.tap(find.byKey(const ValueKey('create-light-command')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('light-option-point')));
      await tester.pump();

      expect(mockApi.createLightCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._lightSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets(
    'routes create sculpt through the Rust facade and refreshes snapshot state',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('create-sculpt-command')),
        200,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.pump();
      await tester.tap(find.byKey(const ValueKey('create-sculpt-command')));
      await tester.pump();

      expect(mockApi.createSculptCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._sculptSnapshot);
      expect(requestFrameCalls, greaterThan(0));
    },
  );

  testWidgets('routes redo through the Rust facade and refreshes snapshot state', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.tap(find.text('Box'));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('undo-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('undo-command')));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('redo-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('redo-command')));
    await tester.pump();

    expect(mockApi.addBoxCalls, 1);
    expect(mockApi.undoCalls, 1);
    expect(mockApi.redoCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedUndoSnapshot);
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('routes viewport gestures to native texture commands', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));
    final hoverGesture = await tester.createGesture(
      kind: PointerDeviceKind.mouse,
    );
    await hoverGesture.addPointer(location: viewportCenter);
    await hoverGesture.moveTo(viewportCenter);
    await tester.pump();
    await hoverGesture.moveTo(viewportCenter + const Offset(16, 8));
    await tester.pump();
    await hoverGesture.moveTo(const Offset(0, 0));
    await tester.pump();

    await tester.tapAt(viewportCenter);
    await tester.pump();

    final orbitGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await orbitGesture.moveBy(const Offset(24, 12));
    await orbitGesture.up();
    await tester.pump();

    final panGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.mouse,
      buttons: kSecondaryMouseButton,
    );
    await panGesture.moveBy(const Offset(18, 10));
    await panGesture.up();
    await tester.pump();

    await tester.sendEventToBinding(
      PointerScrollEvent(
        position: viewportCenter,
        scrollDelta: const Offset(0, -24),
      ),
    );
    await tester.pump();
    await hoverGesture.removePointer();

    expect(pickCalls, 1);
    expect(hoverCalls, greaterThan(0));
    expect(clearHoverCalls, greaterThan(0));
    expect(orbitCalls, greaterThan(0));
    expect(panCalls, greaterThan(0));
    expect(zoomCalls, 1);
  });

  testWidgets('viewport gesture thresholds separate touch taps from drags', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));

    final tapGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.touch,
    );
    await tapGesture.moveBy(const Offset(10, 0));
    await tapGesture.up();
    await tester.pump();

    expect(pickCalls, 1);
    expect(orbitCalls, 0);

    final dragGesture = await tester.startGesture(
      viewportCenter,
      kind: PointerDeviceKind.touch,
    );
    await dragGesture.moveBy(const Offset(20, 0));
    await tester.pump();
    await dragGesture.moveBy(const Offset(20, 10));
    await dragGesture.up();
    await tester.pump();

    expect(orbitCalls, greaterThan(0));
    expect(pickCalls, 1);
  });

  testWidgets('routes inspector camera commands through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.text('Focus Selected'),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();

    await tester.tap(find.text('Focus Selected'));
    await tester.pump();
    await tester.tap(find.text('Front'));
    await tester.pump();
    await tester.tap(find.text('Use Ortho'));
    await tester.pump();

    expect(mockApi.focusSelectedCalls, 1);
    expect(mockApi.cameraFrontCalls, 1);
    expect(mockApi.toggleOrthographicCalls, 1);
    expect(find.text('Use Perspective'), findsOneWidget);
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('tablet shell uses stacked panes and modal command surfaces', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);
    await revealTabletBottomSheetContent(
      tester,
      find.byKey(const ValueKey('open-command-panel')),
    );

    expect(find.byType(ShellStackedPaneLayout), findsOneWidget);
    expect(find.byType(ShellDesktopSidePanel), findsNothing);

    await tester.tap(find.byKey(const ValueKey('open-command-panel')));
    await tester.pumpAndSettle();

    expect(find.byType(ShellModalPanel), findsOneWidget);
    expect(find.text('Workspace Commands'), findsOneWidget);

    await tester.tap(find.text('Box'));
    await tester.pumpAndSettle();

    expect(mockApi.addBoxCalls, 1);
    expect(find.byType(ShellModalPanel), findsNothing);
  });

  testWidgets('desktop shell keeps the side panel layout active', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));

    expect(find.byType(ShellDesktopSidePanel), findsOneWidget);
    expect(find.byType(ShellStackedPaneLayout), findsNothing);
    expect(find.byKey(const ValueKey('open-command-panel')), findsNothing);
    expect(find.widgetWithText(FilledButton, 'Sphere'), findsOneWidget);
  });
}


