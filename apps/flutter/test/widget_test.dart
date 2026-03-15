import 'dart:convert';

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
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_event.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

class _MockRustApi extends RustLibApi {
  static const String _baseSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _documentSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":true,"recent_files":["C:\\\\Scenes\\\\hero.sdf","C:\\\\Scenes\\\\blockout.sdf"],"recovery_available":true,"recovery_summary":"Recovered unsaved work found."},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _exportIdleSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":false,"recent_files":["C:\\\\Scenes\\\\hero.sdf"],"recovery_available":false,"recovery_summary":null},"export":{"resolution":128,"min_resolution":16,"max_resolution":2048,"adaptive":false,"presets":[{"name":"Low","resolution":64},{"name":"Medium","resolution":128},{"name":"High","resolution":256},{"name":"Ultra","resolution":512}],"status":{"state":"idle","progress":0,"total":0,"resolution":128,"phase_label":null,"target_file_name":null,"target_file_path":null,"format_label":null,"message":null,"is_error":false}},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _exportResolutionSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":false,"recent_files":["C:\\\\Scenes\\\\hero.sdf"],"recovery_available":false,"recovery_summary":null},"export":{"resolution":256,"min_resolution":16,"max_resolution":2048,"adaptive":false,"presets":[{"name":"Low","resolution":64},{"name":"Medium","resolution":128},{"name":"High","resolution":256},{"name":"Ultra","resolution":512}],"status":{"state":"idle","progress":0,"total":0,"resolution":256,"phase_label":null,"target_file_name":null,"target_file_path":null,"format_label":null,"message":null,"is_error":false}},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _exportAdaptiveSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":false,"recent_files":["C:\\\\Scenes\\\\hero.sdf"],"recovery_available":false,"recovery_summary":null},"export":{"resolution":256,"min_resolution":16,"max_resolution":2048,"adaptive":true,"presets":[{"name":"Low","resolution":64},{"name":"Medium","resolution":128},{"name":"High","resolution":256},{"name":"Ultra","resolution":512}],"status":{"state":"idle","progress":0,"total":0,"resolution":256,"phase_label":null,"target_file_name":null,"target_file_path":null,"format_label":null,"message":null,"is_error":false}},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _exportRunningSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":false,"recent_files":["C:\\\\Scenes\\\\hero.sdf"],"recovery_available":false,"recovery_summary":null},"export":{"resolution":256,"min_resolution":16,"max_resolution":2048,"adaptive":true,"presets":[{"name":"Low","resolution":64},{"name":"Medium","resolution":128},{"name":"High","resolution":256},{"name":"Ultra","resolution":512}],"status":{"state":"in_progress","progress":8,"total":33,"resolution":16,"phase_label":"Phase 1/3: Sampling SDF field (8/17)","target_file_name":"hero.obj","target_file_path":"C:\\\\Exports\\\\hero.obj","format_label":"OBJ","message":null,"is_error":false}},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _exportDoneSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"document":{"current_file_path":"C:\\\\Scenes\\\\hero.sdf","current_file_name":"hero.sdf","has_unsaved_changes":false,"recent_files":["C:\\\\Scenes\\\\hero.sdf"],"recovery_available":false,"recovery_summary":null},"export":{"resolution":256,"min_resolution":16,"max_resolution":2048,"adaptive":true,"presets":[{"name":"Low","resolution":64},{"name":"Medium","resolution":128},{"name":"High","resolution":256},{"name":"Ultra","resolution":512}],"status":{"state":"idle","progress":0,"total":0,"resolution":256,"phase_label":null,"target_file_name":null,"target_file_path":null,"format_label":null,"message":"Exported OBJ (128 verts, 64 tris)","is_error":false}},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
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
  static const String _selectedTransformManipulatorSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true,"manipulator_mode_id":"translate","manipulator_mode_label":"Move","manipulator_space_id":"local","manipulator_space_label":"Local","manipulator_visible":true,"can_reset_pivot":false,"pivot_offset":{"x":0.0,"y":0.0,"z":0.0}}}''';
  static const String _selectedTransformRotateSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true,"manipulator_mode_id":"rotate","manipulator_mode_label":"Rotate","manipulator_space_id":"local","manipulator_space_label":"Local","manipulator_visible":true,"can_reset_pivot":false,"pivot_offset":{"x":0.0,"y":0.0,"z":0.0}}}''';
  static const String _selectedTransformRotateWorldSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true,"manipulator_mode_id":"rotate","manipulator_mode_label":"Rotate","manipulator_space_id":"world","manipulator_space_label":"World","manipulator_visible":true,"can_reset_pivot":false,"pivot_offset":{"x":0.0,"y":0.0,"z":0.0}}}''';
  static const String _selectedTransformPivotSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true,"manipulator_mode_id":"rotate","manipulator_mode_label":"Rotate","manipulator_space_id":"world","manipulator_space_label":"World","manipulator_visible":true,"can_reset_pivot":true,"pivot_offset":{"x":0.25,"y":0.0,"z":0.0}}}''';
  static const String _selectedTransformPivotResetSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":0.0,"y":0.0,"z":0.0},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true,"manipulator_mode_id":"rotate","manipulator_mode_label":"Rotate","manipulator_space_id":"world","manipulator_space_label":"World","manipulator_visible":true,"can_reset_pivot":false,"pivot_offset":{"x":0.0,"y":0.0,"z":0.0}}}''';
  static const String _selectedTransformMovedSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":0.0,"y":0.0,"z":0.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":28,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformRotatedSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":10.0,"y":20.0,"z":30.0},"scale":{"x":1.0,"y":1.0,"z":1.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":29,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _selectedTransformScaledSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"selected_node_properties":{"node_id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"transform":{"position_label":"Translation","position":{"x":1.0,"y":-2.0,"z":3.5},"rotation_degrees":{"x":10.0,"y":20.0,"z":30.0},"scale":{"x":0.01,"y":2.0,"z":100.0}},"primitive":null,"material":null},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":30,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _modifierSnapshot = '''{"selected_node":{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Twist","kind_label":"Twist","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":1,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _lightSnapshot = '''{"selected_node":{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Transform 4","kind_label":"Transform","visible":true,"locked":false,"children":[{"id":9,"name":"Point","kind_label":"Point","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":9,"visible_nodes":9,"top_level_nodes":5,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":4,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":4,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":3.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _sculptSnapshot = '''{"selected_node":{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false},"top_level_nodes":[{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false}],"scene_tree_roots":[{"id":8,"name":"Sculpt","kind_label":"Sculpt","visible":true,"locked":false,"children":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false,"children":[]}]}],"history":{"can_undo":true,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":8,"visible_nodes":8,"top_level_nodes":4,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":1,"light_nodes":3,"voxel_memory_bytes":1048576,"sdf_eval_complexity":2,"structure_key":12,"data_fingerprint":24,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _baseRedoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":true},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';
  static const String _orthoSnapshot = '''{"selected_node":null,"top_level_nodes":[{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false}],"history":{"can_undo":false,"can_redo":false},"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":true,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"stats":{"total_nodes":7,"visible_nodes":7,"top_level_nodes":1,"primitive_nodes":1,"operation_nodes":0,"transform_nodes":3,"modifier_nodes":0,"sculpt_nodes":0,"light_nodes":3,"voxel_memory_bytes":0,"sdf_eval_complexity":1,"structure_key":11,"data_fingerprint":22,"bounds_min":{"x":-2.5,"y":-2.5,"z":-2.5},"bounds_max":{"x":2.5,"y":2.5,"z":2.5}},"tool":{"active_tool_label":"Select","shading_mode_label":"Full","grid_enabled":true}}''';

  static String _withFields(String baseJson, Map<String, Object?> fields) {
    final snapshot = jsonDecode(baseJson) as Map<String, dynamic>;
    snapshot.addAll(fields);
    return jsonEncode(snapshot);
  }

  static Map<String, Object?> _idleImport() => <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'idle',
          'progress': 0,
          'total': 0,
          'filename': null,
          'phase_label': null,
          'message': null,
          'is_error': false,
        },
      };

  static Map<String, Object?> _idleSculptConvert() => <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'idle',
          'progress': 0,
          'total': 0,
          'target_name': null,
          'phase_label': null,
          'message': null,
          'is_error': false,
        },
      };

  static Map<String, Object?> _sculptPayload({
    required bool canResumeSelected,
    required bool canStop,
    required int desiredResolution,
    String? brushModeId,
    String? brushModeLabel,
    double? brushRadius,
    double? brushStrength,
    String? symmetryAxisId,
    String? symmetryAxisLabel,
  }) {
    final session =
        brushModeId == null
            ? null
            : <String, Object?>{
                'node_id': 8,
                'node_name': 'Sculpt',
                'brush_mode_id': brushModeId,
                'brush_mode_label': brushModeLabel!,
                'brush_radius': brushRadius!,
                'brush_strength': brushStrength!,
                'symmetry_axis_id': symmetryAxisId!,
                'symmetry_axis_label': symmetryAxisLabel!,
              };
    return <String, Object?>{
      'selected': <String, Object?>{
        'node_id': 8,
        'node_name': 'Sculpt',
        'current_resolution': 64,
        'desired_resolution': desiredResolution,
      },
      'session': session,
      'can_resume_selected': canResumeSelected,
      'can_stop': canStop,
      'max_resolution': 256,
    };
  }

  static final String _importDialogSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{
      'import': <String, Object?>{
        'dialog': <String, Object?>{
          'filename': 'hero_mesh.obj',
          'resolution': 96,
          'auto_resolution': 96,
          'use_auto': true,
          'vertex_count': 2048,
          'triangle_count': 4096,
          'bounds_size': <String, double>{'x': 3.0, 'y': 2.0, 'z': 1.5},
          'min_resolution': 16,
          'max_resolution': 512,
        },
        'status': _idleImport()['status'],
      },
    },
  );

  static final String _importManualSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{
      'import': <String, Object?>{
        'dialog': <String, Object?>{
          'filename': 'hero_mesh.obj',
          'resolution': 96,
          'auto_resolution': 96,
          'use_auto': false,
          'vertex_count': 2048,
          'triangle_count': 4096,
          'bounds_size': <String, double>{'x': 3.0, 'y': 2.0, 'z': 1.5},
          'min_resolution': 16,
          'max_resolution': 512,
        },
        'status': _idleImport()['status'],
      },
    },
  );

  static final String _importResolutionSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{
      'import': <String, Object?>{
        'dialog': <String, Object?>{
          'filename': 'hero_mesh.obj',
          'resolution': 144,
          'auto_resolution': 96,
          'use_auto': false,
          'vertex_count': 2048,
          'triangle_count': 4096,
          'bounds_size': <String, double>{'x': 3.0, 'y': 2.0, 'z': 1.5},
          'min_resolution': 16,
          'max_resolution': 512,
        },
        'status': _idleImport()['status'],
      },
    },
  );

  static final String _importRunningSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{
      'import': <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'in_progress',
          'progress': 24,
          'total': 144,
          'filename': 'hero_mesh.obj',
          'phase_label': 'Voxelizing mesh...',
          'message': null,
          'is_error': false,
        },
      },
    },
  );

  static final String _importDoneSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{
      'import': <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'idle',
          'progress': 0,
          'total': 0,
          'filename': null,
          'phase_label': null,
          'message': 'Imported hero_mesh.obj as sculpt geometry',
          'is_error': false,
        },
      },
    },
  );

  static final String _sculptConvertDialogSnapshot = _withFields(
    _selectedSnapshot,
    <String, Object?>{
      'sculpt_convert': <String, Object?>{
        'dialog': <String, Object?>{
          'target_node_id': 1,
          'target_name': 'Sphere',
          'mode_id': 'active_node',
          'mode_label': 'Bake active node',
          'resolution': 64,
          'min_resolution': 16,
          'max_resolution': 256,
        },
        'status': _idleSculptConvert()['status'],
      },
    },
  );

  static final String _sculptConvertFlattenSnapshot = _withFields(
    _selectedSnapshot,
    <String, Object?>{
      'sculpt_convert': <String, Object?>{
        'dialog': <String, Object?>{
          'target_node_id': 1,
          'target_name': 'Sphere',
          'mode_id': 'whole_scene_flatten',
          'mode_label': 'Bake whole scene + flatten',
          'resolution': 64,
          'min_resolution': 16,
          'max_resolution': 256,
        },
        'status': _idleSculptConvert()['status'],
      },
    },
  );

  static final String _sculptConvertResolutionSnapshot = _withFields(
    _selectedSnapshot,
    <String, Object?>{
      'sculpt_convert': <String, Object?>{
        'dialog': <String, Object?>{
          'target_node_id': 1,
          'target_name': 'Sphere',
          'mode_id': 'whole_scene_flatten',
          'mode_label': 'Bake whole scene + flatten',
          'resolution': 96,
          'min_resolution': 16,
          'max_resolution': 256,
        },
        'status': _idleSculptConvert()['status'],
      },
    },
  );

  static final String _sculptConvertRunningSnapshot = _withFields(
    _selectedSnapshot,
    <String, Object?>{
      'sculpt_convert': <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'in_progress',
          'progress': 19,
          'total': 96,
          'target_name': 'Sphere',
          'phase_label': 'Preparing sculpt volume...',
          'message': null,
          'is_error': false,
        },
      },
    },
  );

  static final String _sculptConvertDoneSnapshot = _withFields(
    _selectedSnapshot,
    <String, Object?>{
      'sculpt_convert': <String, Object?>{
        'dialog': null,
        'status': <String, Object?>{
          'state': 'idle',
          'progress': 0,
          'total': 0,
          'target_name': null,
          'phase_label': null,
          'message': 'Converted Sphere to sculpt',
          'is_error': false,
        },
      },
    },
  );

  static final String _sculptActiveSnapshot = _withFields(
    _sculptSnapshot,
    <String, Object?>{
      'tool': <String, Object?>{
        'active_tool_label': 'Sculpt',
        'shading_mode_label': 'Full',
        'grid_enabled': true,
      },
      'sculpt': _sculptPayload(
        canResumeSelected: false,
        canStop: true,
        desiredResolution: 64,
        brushModeId: 'add',
        brushModeLabel: 'Add',
        brushRadius: 0.15,
        brushStrength: 0.05,
        symmetryAxisId: 'off',
        symmetryAxisLabel: 'Off',
      ),
    },
  );

  static final String _sculptStoppedSnapshot = _withFields(
    _sculptSnapshot,
    <String, Object?>{
      'tool': <String, Object?>{
        'active_tool_label': 'Select',
        'shading_mode_label': 'Full',
        'grid_enabled': true,
      },
      'sculpt': _sculptPayload(
        canResumeSelected: true,
        canStop: false,
        desiredResolution: 64,
      ),
    },
  );

  static final String _sculptGrabSnapshot = _withFields(
    _sculptSnapshot,
    <String, Object?>{
      'tool': <String, Object?>{
        'active_tool_label': 'Sculpt',
        'shading_mode_label': 'Full',
        'grid_enabled': true,
      },
      'sculpt': _sculptPayload(
        canResumeSelected: false,
        canStop: true,
        desiredResolution: 64,
        brushModeId: 'grab',
        brushModeLabel: 'Grab',
        brushRadius: 0.15,
        brushStrength: 3.0,
        symmetryAxisId: 'off',
        symmetryAxisLabel: 'Off',
      ),
    },
  );

  static final String _sculptSymmetrySnapshot = _withFields(
    _sculptSnapshot,
    <String, Object?>{
      'tool': <String, Object?>{
        'active_tool_label': 'Sculpt',
        'shading_mode_label': 'Full',
        'grid_enabled': true,
      },
      'sculpt': _sculptPayload(
        canResumeSelected: false,
        canStop: true,
        desiredResolution: 64,
        brushModeId: 'grab',
        brushModeLabel: 'Grab',
        brushRadius: 0.15,
        brushStrength: 3.0,
        symmetryAxisId: 'z',
        symmetryAxisLabel: 'Z',
      ),
    },
  );

  static final String _sculptResolutionSnapshot = _withFields(
    _sculptSnapshot,
    <String, Object?>{
      'tool': <String, Object?>{
        'active_tool_label': 'Sculpt',
        'shading_mode_label': 'Full',
        'grid_enabled': true,
      },
      'sculpt': _sculptPayload(
        canResumeSelected: false,
        canStop: true,
        desiredResolution: 128,
        brushModeId: 'grab',
        brushModeLabel: 'Grab',
        brushRadius: 0.15,
        brushStrength: 3.0,
        symmetryAxisId: 'z',
        symmetryAxisLabel: 'Z',
      ),
    },
  );

  String currentSnapshot = _baseSnapshot;
  int sceneSnapshotJsonCalls = 0;
  int addBoxCalls = 0;
  int createOperationCalls = 0;
  int createTransformCalls = 0;
  int createModifierCalls = 0;
  int createLightCalls = 0;
  int createSculptCalls = 0;
  int resumeSculptingSelectedCalls = 0;
  int stopSculptingCalls = 0;
  int setSculptBrushModeCalls = 0;
  int setSculptBrushRadiusCalls = 0;
  int setSculptBrushStrengthCalls = 0;
  int setSculptSymmetryAxisCalls = 0;
  int setSelectedSculptResolutionCalls = 0;
  int duplicateSelectedCalls = 0;
  int newSceneCalls = 0;
  int openSceneCalls = 0;
  int openRecentSceneCalls = 0;
  int saveSceneCalls = 0;
  int saveSceneAsCalls = 0;
  int recoverAutosaveCalls = 0;
  int discardRecoveryCalls = 0;
  int setExportResolutionCalls = 0;
  int setAdaptiveExportCalls = 0;
  int startExportCalls = 0;
  int cancelExportCalls = 0;
  int openImportDialogCalls = 0;
  int cancelImportDialogCalls = 0;
  int setImportUseAutoCalls = 0;
  int setImportResolutionCalls = 0;
  int startImportCalls = 0;
  int cancelImportCalls = 0;
  int openSculptConvertDialogCalls = 0;
  int cancelSculptConvertDialogCalls = 0;
  int setSculptConvertModeCalls = 0;
  int setSculptConvertResolutionCalls = 0;
  int startSculptConvertCalls = 0;
  int renameNodeCalls = 0;
  int setSelectedPrimitiveParameterCalls = 0;
  int setSelectedMaterialFloatCalls = 0;
  int setSelectedMaterialColorCalls = 0;
  int setSelectedTransformPositionCalls = 0;
  int setSelectedTransformRotationDegreesCalls = 0;
  int setSelectedTransformScaleCalls = 0;
  int setManipulatorModeCalls = 0;
  int toggleManipulatorSpaceCalls = 0;
  int nudgeManipulatorPivotOffsetCalls = 0;
  int resetManipulatorPivotCalls = 0;
  int nudgeSelectedTranslationCalls = 0;
  int nudgeSelectedRotationDegreesCalls = 0;
  int nudgeSelectedScaleCalls = 0;
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
    sceneSnapshotJsonCalls = 0;
    addBoxCalls = 0;
    createOperationCalls = 0;
    createTransformCalls = 0;
    createModifierCalls = 0;
    createLightCalls = 0;
    createSculptCalls = 0;
    resumeSculptingSelectedCalls = 0;
    stopSculptingCalls = 0;
    setSculptBrushModeCalls = 0;
    setSculptBrushRadiusCalls = 0;
    setSculptBrushStrengthCalls = 0;
    setSculptSymmetryAxisCalls = 0;
    setSelectedSculptResolutionCalls = 0;
    duplicateSelectedCalls = 0;
    newSceneCalls = 0;
    openSceneCalls = 0;
    openRecentSceneCalls = 0;
    saveSceneCalls = 0;
    saveSceneAsCalls = 0;
    recoverAutosaveCalls = 0;
    discardRecoveryCalls = 0;
    setExportResolutionCalls = 0;
    setAdaptiveExportCalls = 0;
    startExportCalls = 0;
    cancelExportCalls = 0;
    openImportDialogCalls = 0;
    cancelImportDialogCalls = 0;
    setImportUseAutoCalls = 0;
    setImportResolutionCalls = 0;
    startImportCalls = 0;
    cancelImportCalls = 0;
    openSculptConvertDialogCalls = 0;
    cancelSculptConvertDialogCalls = 0;
    setSculptConvertModeCalls = 0;
    setSculptConvertResolutionCalls = 0;
    startSculptConvertCalls = 0;
    renameNodeCalls = 0;
    setSelectedPrimitiveParameterCalls = 0;
    setSelectedMaterialFloatCalls = 0;
    setSelectedMaterialColorCalls = 0;
    setSelectedTransformPositionCalls = 0;
    setSelectedTransformRotationDegreesCalls = 0;
    setSelectedTransformScaleCalls = 0;
    setManipulatorModeCalls = 0;
    toggleManipulatorSpaceCalls = 0;
    nudgeManipulatorPivotOffsetCalls = 0;
    resetManipulatorPivotCalls = 0;
    nudgeSelectedTranslationCalls = 0;
    nudgeSelectedRotationDegreesCalls = 0;
    nudgeSelectedScaleCalls = 0;
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
  String crateApiSimpleDiscardRecovery() {
    discardRecoveryCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetExportResolution({required int resolution}) {
    setExportResolutionCalls += 1;
    currentSnapshot = _exportResolutionSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetAdaptiveExport({required bool enabled}) {
    setAdaptiveExportCalls += 1;
    currentSnapshot = enabled ? _exportAdaptiveSnapshot : _exportResolutionSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleStartExport() {
    startExportCalls += 1;
    currentSnapshot = _exportRunningSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCancelExport() {
    cancelExportCalls += 1;
    currentSnapshot = _exportIdleSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleOpenImportDialog() {
    openImportDialogCalls += 1;
    currentSnapshot = _importDialogSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCancelImportDialog() {
    cancelImportDialogCalls += 1;
    currentSnapshot = _baseSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetImportUseAuto({required bool useAuto}) {
    setImportUseAutoCalls += 1;
    currentSnapshot = useAuto ? _importDialogSnapshot : _importManualSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetImportResolution({required int resolution}) {
    setImportResolutionCalls += 1;
    currentSnapshot = _importResolutionSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleStartImport() {
    startImportCalls += 1;
    currentSnapshot = _importRunningSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCancelImport() {
    cancelImportCalls += 1;
    currentSnapshot = _baseSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleOpenSculptConvertDialogForSelected() {
    openSculptConvertDialogCalls += 1;
    currentSnapshot = _sculptConvertDialogSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleCancelSculptConvertDialog() {
    cancelSculptConvertDialogCalls += 1;
    currentSnapshot = _selectedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptConvertMode({required String modeId}) {
    setSculptConvertModeCalls += 1;
    currentSnapshot = modeId == 'whole_scene_flatten'
        ? _sculptConvertFlattenSnapshot
        : _sculptConvertDialogSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptConvertResolution({required int resolution}) {
    setSculptConvertResolutionCalls += 1;
    currentSnapshot = _sculptConvertResolutionSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleStartSculptConvert() {
    startSculptConvertCalls += 1;
    currentSnapshot = _sculptConvertRunningSnapshot;
    return currentSnapshot;
  }

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
    currentSnapshot = _sculptActiveSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleResumeSculptingSelected() {
    resumeSculptingSelectedCalls += 1;
    currentSnapshot = _sculptActiveSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleStopSculpting() {
    stopSculptingCalls += 1;
    currentSnapshot = _sculptStoppedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptBrushMode({required String modeId}) {
    setSculptBrushModeCalls += 1;
    currentSnapshot = modeId == 'grab' ? _sculptGrabSnapshot : _sculptActiveSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptBrushRadius({required double radius}) {
    setSculptBrushRadiusCalls += 1;
    currentSnapshot = _sculptActiveSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptBrushStrength({required double strength}) {
    setSculptBrushStrengthCalls += 1;
    currentSnapshot = _sculptGrabSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSculptSymmetryAxis({required String axisId}) {
    setSculptSymmetryAxisCalls += 1;
    currentSnapshot = axisId == 'z' ? _sculptSymmetrySnapshot : _sculptActiveSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSetSelectedSculptResolution({required int resolution}) {
    setSelectedSculptResolutionCalls += 1;
    currentSnapshot = _sculptResolutionSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleDeleteSelected() {
    deleteSelectedCalls += 1;
    currentSnapshot = _baseUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleNewScene() {
    newSceneCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleOpenRecentScene({required String path}) {
    openRecentSceneCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleOpenScene() {
    openSceneCalls += 1;
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
  String crateApiSimpleSetManipulatorMode({required String modeId}) {
    setManipulatorModeCalls += 1;
    currentSnapshot = switch (modeId) {
      'rotate' => _selectedTransformRotateSnapshot,
      _ => _selectedTransformManipulatorSnapshot,
    };
    return currentSnapshot;
  }

  @override
  String crateApiSimpleToggleManipulatorSpace() {
    toggleManipulatorSpaceCalls += 1;
    currentSnapshot = _selectedTransformRotateWorldSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleNudgeManipulatorPivotOffset({
    required double x,
    required double y,
    required double z,
  }) {
    nudgeManipulatorPivotOffsetCalls += 1;
    currentSnapshot = _selectedTransformPivotSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleResetManipulatorPivot() {
    resetManipulatorPivotCalls += 1;
    currentSnapshot = _selectedTransformPivotResetSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleNudgeSelectedTranslation({
    required double deltaX,
    required double deltaY,
    required double deltaZ,
  }) {
    nudgeSelectedTranslationCalls += 1;
    currentSnapshot = _selectedTransformMovedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleNudgeSelectedRotationDegrees({
    required double deltaXDegrees,
    required double deltaYDegrees,
    required double deltaZDegrees,
  }) {
    nudgeSelectedRotationDegreesCalls += 1;
    currentSnapshot = _selectedTransformRotatedSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleNudgeSelectedScale({
    required double deltaX,
    required double deltaY,
    required double deltaZ,
  }) {
    nudgeSelectedScaleCalls += 1;
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
  String crateApiSimpleRecoverAutosave() {
    recoverAutosaveCalls += 1;
    return currentSnapshot;
  }

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
  String crateApiSimpleSaveScene() {
    saveSceneCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSaveSceneAs() {
    saveSceneAsCalls += 1;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleRedo() {
    redoCalls += 1;
    currentSnapshot = _selectedUndoSnapshot;
    return currentSnapshot;
  }

  @override
  String crateApiSimpleSceneSnapshotJson() {
    sceneSnapshotJsonCalls += 1;
    if (currentSnapshot == _exportRunningSnapshot && sceneSnapshotJsonCalls >= 3) {
      currentSnapshot = _exportDoneSnapshot;
    } else if (currentSnapshot == _importRunningSnapshot &&
        sceneSnapshotJsonCalls >= 3) {
      currentSnapshot = _importDoneSnapshot;
    } else if (currentSnapshot == _sculptConvertRunningSnapshot &&
        sceneSnapshotJsonCalls >= 3) {
      currentSnapshot = _sculptConvertDoneSnapshot;
    }
    return currentSnapshot;
  }

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

  Future<void> dispatchTextureEvent(
    WidgetTester tester,
    Map<Object?, Object?> event,
  ) async {
    final dynamic state = tester.state(find.byType(BridgeStatusPage));
    state.debugHandleTextureEvent(TextureViewportEvent.fromDynamic(event));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 50));
  }

  Map<Object?, Object?> buildTextureEvent({
    int frameCount = 1,
    double frameTimeMs = 16.0,
    int droppedFrameCount = 0,
    String interactionPhase = 'idle',
    bool sceneStateChanged = false,
    String feedbackJson = '',
    String? hostError,
  }) {
    return <Object?, Object?>{
      'textureId': 7,
      'frameWidth': 640,
      'frameHeight': 360,
      'frameTimeMs': frameTimeMs,
      'frameCount': frameCount,
      'droppedFrameCount': droppedFrameCount,
      'interactionPhase': interactionPhase,
      'sceneStateChanged': sceneStateChanged,
      'feedbackJson': feedbackJson,
      if (hostError case final String error) 'hostError': error,
    };
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

  testWidgets('routes viewport tool overlay commands through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedTransformManipulatorSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.tap(find.byKey(const ValueKey('viewport-tool-mode-rotate')));
    await tester.pump();

    expect(mockApi.setManipulatorModeCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformRotateSnapshot);

    await tester.tap(find.byKey(const ValueKey('viewport-tool-space-toggle')));
    await tester.pump();

    expect(mockApi.toggleManipulatorSpaceCalls, 1);
    expect(
      mockApi.currentSnapshot,
      _MockRustApi._selectedTransformRotateWorldSnapshot,
    );

    await tester.tap(find.byKey(const ValueKey('viewport-tool-pivot-x-positive')));
    await tester.pump();

    expect(mockApi.nudgeManipulatorPivotOffsetCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformPivotSnapshot);

    await tester.tap(find.byKey(const ValueKey('viewport-tool-pivot-reset')));
    await tester.pump();

    expect(mockApi.resetManipulatorPivotCalls, 1);
    expect(
      mockApi.currentSnapshot,
      _MockRustApi._selectedTransformPivotResetSnapshot,
    );

    await tester.tap(find.byKey(const ValueKey('viewport-tool-nudge-x-positive')));
    await tester.pump();

    expect(mockApi.nudgeSelectedRotationDegreesCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformRotatedSnapshot);
    expect(requestFrameCalls, greaterThan(0));
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
          hostError: null,
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

  testWidgets('routes document lifecycle commands through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._documentSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('document-new-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('document-new-command')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('document-open-command')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('document-save-command')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('document-save-as-command')));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('document-recover-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('document-recover-command')));
    await tester.pump();
    await tester.ensureVisible(
      find.byKey(const ValueKey('document-discard-recovery-command')),
    );
    await tester.pump();
    await tester.tap(
      find.byKey(const ValueKey('document-discard-recovery-command')),
    );
    await tester.pump();
    await tester.ensureVisible(find.text('blockout.sdf'));
    await tester.pump();
    await tester.tap(find.text('blockout.sdf'));
    await tester.pump();

    expect(mockApi.newSceneCalls, 1);
    expect(mockApi.openSceneCalls, 1);
    expect(mockApi.saveSceneCalls, 1);
    expect(mockApi.saveSceneAsCalls, 1);
    expect(mockApi.recoverAutosaveCalls, 1);
    expect(mockApi.discardRecoveryCalls, 1);
    expect(mockApi.openRecentSceneCalls, 1);
    expect(requestFrameCalls, greaterThan(0));
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

  testWidgets('routes export settings and export start through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._exportIdleSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('export-resolution-field')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.enterText(
      find.byKey(const ValueKey('export-resolution-field')),
      '256',
    );
    await tester.tap(find.byKey(const ValueKey('export-apply-resolution')));
    await tester.pump();

    await tester.tap(find.byKey(const ValueKey('export-adaptive-toggle')));
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('export-start-command')));
    await tester.pump();

    expect(mockApi.setExportResolutionCalls, 1);
    expect(mockApi.setAdaptiveExportCalls, 1);
    expect(mockApi.startExportCalls, 1);
    expect(find.byKey(const ValueKey('export-progress-indicator')), findsOneWidget);
    expect(find.textContaining('Exporting hero.obj'), findsOneWidget);

    await tester.pump(const Duration(milliseconds: 500));
    await tester.pumpAndSettle();

    expect(mockApi.sceneSnapshotJsonCalls, greaterThanOrEqualTo(3));
    expect(find.text('Exported OBJ (128 verts, 64 tris)'), findsOneWidget);
    expect(find.byKey(const ValueKey('export-progress-indicator')), findsNothing);
  });

  testWidgets('routes import settings and import start through the Rust facade', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('open-import-dialog-command')),
      200,
      scrollable: find.byType(Scrollable).last,
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('open-import-dialog-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Manual'));
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byKey(const ValueKey('import-resolution-field')),
      '144',
    );
    await tester.tap(find.byKey(const ValueKey('import-apply-resolution')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('start-import-command')));
    await tester.pump();

    expect(mockApi.openImportDialogCalls, 1);
    expect(mockApi.setImportUseAutoCalls, 1);
    expect(mockApi.setImportResolutionCalls, 1);
    expect(mockApi.startImportCalls, 1);
    expect(find.byKey(const ValueKey('import-progress-indicator')), findsOneWidget);
    expect(find.textContaining('Importing hero_mesh.obj'), findsOneWidget);

    await tester.pump(const Duration(milliseconds: 500));
    await tester.pumpAndSettle();

    expect(mockApi.sceneSnapshotJsonCalls, greaterThanOrEqualTo(3));
    expect(find.text('Imported hero_mesh.obj as sculpt geometry'), findsOneWidget);
    expect(find.byKey(const ValueKey('import-progress-indicator')), findsNothing);
  });

  testWidgets(
    'routes sculpt convert settings and start through the Rust facade',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('open-sculpt-convert-dialog-command')),
        200,
        scrollable: find.byType(Scrollable).last,
      );
      await tester.pumpAndSettle();

      await tester.tap(
        find.byKey(const ValueKey('open-sculpt-convert-dialog-command')),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.text('Bake whole scene + flatten'));
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const ValueKey('sculpt-convert-resolution-field')),
        '96',
      );
      await tester.tap(
        find.byKey(const ValueKey('sculpt-convert-apply-resolution')),
      );
      await tester.pump();
      await tester.tap(find.byKey(const ValueKey('start-sculpt-convert-command')));
      await tester.pump();

      expect(mockApi.openSculptConvertDialogCalls, 1);
      expect(mockApi.setSculptConvertModeCalls, 1);
      expect(mockApi.setSculptConvertResolutionCalls, 1);
      expect(mockApi.startSculptConvertCalls, 1);
      expect(
        find.byKey(const ValueKey('sculpt-convert-progress-indicator')),
        findsOneWidget,
      );
      expect(find.textContaining('Converting Sphere'), findsOneWidget);

      await tester.pump(const Duration(milliseconds: 500));
      await tester.pumpAndSettle();

      expect(mockApi.sceneSnapshotJsonCalls, greaterThanOrEqualTo(3));
      expect(find.text('Converted Sphere to sculpt'), findsOneWidget);
      expect(
        find.byKey(const ValueKey('sculpt-convert-progress-indicator')),
        findsNothing,
      );
    },
  );

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

  testWidgets('routes sculpt workflow controls through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._sculptActiveSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('sculpt-brush-mode-grab')),
      200,
      scrollable: find.byWidgetPredicate(
        (widget) => widget is Scrollable && widget.axisDirection == AxisDirection.down,
      ).first,
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('sculpt-brush-mode-grab')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const ValueKey('sculpt-radius-increase')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('sculpt-radius-increase')));
    await tester.pump();
    await tester.ensureVisible(find.byKey(const ValueKey('sculpt-strength-increase')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('sculpt-strength-increase')));
    await tester.pump();
    await tester.ensureVisible(find.byKey(const ValueKey('sculpt-symmetry-z')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('sculpt-symmetry-z')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('selected-sculpt-resolution-field')),
    );
    await tester.pumpAndSettle();
    await tester.enterText(
      find.byKey(const ValueKey('selected-sculpt-resolution-field')),
      '128',
    );
    await tester.ensureVisible(
      find.byKey(const ValueKey('selected-sculpt-apply-resolution')),
    );
    await tester.pumpAndSettle();
    await tester.tap(
      find.byKey(const ValueKey('selected-sculpt-apply-resolution')),
    );
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const ValueKey('stop-sculpt-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('stop-sculpt-command')));
    await tester.pumpAndSettle();

    expect(find.byKey(const ValueKey('resume-sculpt-command')), findsOneWidget);

    await tester.ensureVisible(find.byKey(const ValueKey('resume-sculpt-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('resume-sculpt-command')));
    await tester.pumpAndSettle();

    expect(mockApi.setSculptBrushModeCalls, 1);
    expect(mockApi.setSculptBrushRadiusCalls, 1);
    expect(mockApi.setSculptBrushStrengthCalls, 1);
    expect(mockApi.setSculptSymmetryAxisCalls, 1);
    expect(mockApi.setSelectedSculptResolutionCalls, 1);
    expect(mockApi.stopSculptingCalls, 1);
    expect(mockApi.resumeSculptingSelectedCalls, 1);
  });

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
      expect(mockApi.currentSnapshot, _MockRustApi._sculptActiveSnapshot);
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

  testWidgets(
    'texture feedback-only events avoid full scene snapshot refreshes',
    (WidgetTester tester) async {
      await pumpApp(tester);

      expect(mockApi.sceneSnapshotJsonCalls, 1);

      await dispatchTextureEvent(
        tester,
        buildTextureEvent(
          frameCount: 2,
          interactionPhase: 'interacting',
          feedbackJson:
              '{"camera":{"yaw":0.9,"pitch":0.5,"roll":0.0,"distance":4.5,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.4,"y":2.1,"z":3.1}},"selected_node":null,"hovered_node":{"id":99,"name":"Hover Probe","kind_label":"Box","visible":true,"locked":false}}',
        ),
      );

      expect(mockApi.sceneSnapshotJsonCalls, 1);
      expect(find.textContaining('phase interacting'), findsOneWidget);
    },
  );

  testWidgets(
    'texture scene-change events refresh the full scene snapshot once',
    (WidgetTester tester) async {
      await pumpApp(tester);

      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await dispatchTextureEvent(
        tester,
        buildTextureEvent(
          frameCount: 2,
          sceneStateChanged: true,
          feedbackJson:
              '{"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"hovered_node":null}',
        ),
      );

      expect(mockApi.sceneSnapshotJsonCalls, 2);
    },
  );

  testWidgets('surfaces native viewport host errors from texture events', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    await dispatchTextureEvent(
      tester,
      buildTextureEvent(
        frameCount: 2,
        hostError: 'Native viewport frame render failed.',
      ),
    );

    expect(
      find.textContaining('Viewport host error: Native viewport frame render failed.'),
      findsOneWidget,
    );
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

  testWidgets('tablet touch gestures route pan and pinch without selecting', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester);

    final viewportCenter = tester.getCenter(find.byType(ViewportSurface));
    final firstTouch = await tester.createGesture(
      kind: PointerDeviceKind.touch,
    );
    final secondTouch = await tester.createGesture(
      pointer: 7,
      kind: PointerDeviceKind.touch,
    );

    await firstTouch.down(viewportCenter - const Offset(40, 0));
    await secondTouch.down(viewportCenter + const Offset(40, 0));
    await tester.pump();

    await firstTouch.moveTo(viewportCenter - const Offset(60, 0));
    await secondTouch.moveTo(viewportCenter + const Offset(60, 0));
    await tester.pump();

    await firstTouch.moveBy(const Offset(20, 12));
    await secondTouch.moveBy(const Offset(20, 12));
    await tester.pump();

    await firstTouch.up();
    await secondTouch.up();
    await tester.pump();

    expect(zoomCalls, greaterThan(0));
    expect(panCalls, greaterThan(0));
    expect(pickCalls, 0);
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


