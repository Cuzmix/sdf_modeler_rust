import 'dart:convert';

import 'package:flutter/gestures.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sdf_modeler_flutter/app.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/rust/frb_generated.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_desktop_side_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_modal_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_stacked_panes.dart';
import 'package:sdf_modeler_flutter/src/texture/texture_viewport_event.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_feedback_overlay.dart';
import 'package:sdf_modeler_flutter/src/viewport/viewport_surface.dart';

import 'mock_snapshot_adapter.dart';

Finder _commandPanelScrollable() => find.byWidgetPredicate(
  (widget) => widget is Scrollable && widget.axisDirection == AxisDirection.down,
).first;

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

  static final String _advancedLightSnapshot = _withFields(
    _lightSnapshot,
    <String, Object?>{
      'selected_node_properties': <String, Object?>{
        'node_id': 8,
        'name': 'Transform 4',
        'kind_label': 'Transform',
        'visible': true,
        'locked': false,
        'transform': <String, Object?>{
          'position_label': 'Translation',
          'position': <String, double>{'x': 0.0, 'y': 1.5, 'z': 0.0},
          'rotation_degrees': <String, double>{'x': 0.0, 'y': 15.0, 'z': 0.0},
          'scale': <String, double>{'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'primitive': null,
        'material': null,
        'light': <String, Object?>{
          'node_id': 9,
          'transform_node_id': 8,
          'light_type_id': 'spot',
          'light_type_label': 'Spot',
          'color': <String, double>{'x': 0.7, 'y': 0.8, 'z': 0.9},
          'intensity': 2.5,
          'range': 12.0,
          'spot_angle': 35.0,
          'cast_shadows': true,
          'shadow_softness': 8.0,
          'shadow_color': <String, double>{'x': 0.1, 'y': 0.2, 'z': 0.3},
          'volumetric': true,
          'volumetric_density': 0.4,
          'cookie_node_id': 12,
          'cookie_node_name': 'Noise Cookie',
          'cookie_candidates': <Object?>[
            <String, Object?>{
              'node_id': 12,
              'name': 'Noise Cookie',
              'kind_label': 'Plane',
            },
            <String, Object?>{
              'node_id': 13,
              'name': 'Grid Cookie',
              'kind_label': 'Plane',
            },
          ],
          'proximity_mode_id': 'brighten',
          'proximity_mode_label': 'Brighten',
          'proximity_range': 3.0,
          'array_pattern_id': null,
          'array_pattern_label': null,
          'array_count': null,
          'array_radius': null,
          'array_color_variation': null,
          'intensity_expression': 'time * 2.0',
          'intensity_expression_error': null,
          'color_hue_expression': 'sin(time)',
          'color_hue_expression_error': null,
          'supports_range': true,
          'supports_spot_angle': true,
          'supports_shadows': true,
          'supports_volumetric': true,
          'supports_cookie': true,
          'supports_proximity': true,
          'supports_expressions': true,
          'supports_array': false,
        },
      },
      'light_linking': <String, Object?>{
        'lights': <Object?>[
          <String, Object?>{
            'light_node_id': 9,
            'light_name': 'Key Spot',
            'light_type_label': 'Spot',
            'active': true,
            'mask_bit': 0,
            'color': <String, double>{'x': 0.7, 'y': 0.8, 'z': 0.9},
          },
          <String, Object?>{
            'light_node_id': 11,
            'light_name': 'Fill Point',
            'light_type_label': 'Point',
            'active': true,
            'mask_bit': 1,
            'color': <String, double>{'x': 0.5, 'y': 0.4, 'z': 0.7},
          },
        ],
        'geometry_nodes': <Object?>[
          <String, Object?>{
            'node_id': 1,
            'node_name': 'Sphere',
            'kind_label': 'Sphere',
            'light_mask': 1,
          },
          <String, Object?>{
            'node_id': 14,
            'node_name': 'Sculpt',
            'kind_label': 'Sculpt',
            'light_mask': 3,
          },
        ],
        'total_visible_light_count': 2,
        'max_light_count': 8,
      },
    },
  );

  static final String _lightBillboardSnapshot = _withFields(
    _advancedLightSnapshot,
    <String, Object?>{
      'viewport_lights': <Object?>[
        <String, Object?>{
          'light_node_id': 9,
          'transform_node_id': 8,
          'light_type_id': 'spot',
          'light_type_label': 'Spot',
          'world_position': <String, double>{'x': 0.0, 'y': 1.5, 'z': 0.0},
          'direction': <String, double>{'x': 0.0, 'y': -1.0, 'z': 0.0},
          'color': <String, double>{'x': 0.7, 'y': 0.8, 'z': 0.9},
          'intensity': 2.5,
          'range': 12.0,
          'spot_angle': 35.0,
          'active': true,
          'array_positions': <Object?>[],
          'array_colors': <Object?>[],
        },
      ],
    },
  );

  static final String _arrayLightSnapshot = _withFields(
    _advancedLightSnapshot,
    <String, Object?>{
      'selected_node_properties': <String, Object?>{
        'node_id': 8,
        'name': 'Transform 4',
        'kind_label': 'Transform',
        'visible': true,
        'locked': false,
        'transform': <String, Object?>{
          'position_label': 'Translation',
          'position': <String, double>{'x': 0.0, 'y': 1.5, 'z': 0.0},
          'rotation_degrees': <String, double>{'x': 0.0, 'y': 15.0, 'z': 0.0},
          'scale': <String, double>{'x': 1.0, 'y': 1.0, 'z': 1.0},
        },
        'primitive': null,
        'material': null,
        'light': <String, Object?>{
          'node_id': 9,
          'transform_node_id': 8,
          'light_type_id': 'array',
          'light_type_label': 'Array',
          'color': <String, double>{'x': 0.7, 'y': 0.8, 'z': 0.9},
          'intensity': 2.5,
          'range': 12.0,
          'spot_angle': 35.0,
          'cast_shadows': false,
          'shadow_softness': 8.0,
          'shadow_color': <String, double>{'x': 0.1, 'y': 0.2, 'z': 0.3},
          'volumetric': false,
          'volumetric_density': 0.4,
          'cookie_node_id': null,
          'cookie_node_name': null,
          'cookie_candidates': <Object?>[],
          'proximity_mode_id': 'off',
          'proximity_mode_label': 'Off',
          'proximity_range': 3.0,
          'array_pattern_id': 'ring',
          'array_pattern_label': 'Ring',
          'array_count': 6,
          'array_radius': 4.0,
          'array_color_variation': 0.2,
          'intensity_expression': null,
          'intensity_expression_error': null,
          'color_hue_expression': null,
          'color_hue_expression_error': null,
          'supports_range': true,
          'supports_spot_angle': false,
          'supports_shadows': false,
          'supports_volumetric': false,
          'supports_cookie': false,
          'supports_proximity': false,
          'supports_expressions': false,
          'supports_array': true,
        },
      },
    },
  );

  static Map<String, Object?> _defaultRenderPayload() => <String, Object?>{
        'shading_modes': <Map<String, String>>[
          <String, String>{'id': 'full', 'label': 'Full'},
          <String, String>{'id': 'solid', 'label': 'Solid'},
          <String, String>{'id': 'clay', 'label': 'Clay'},
          <String, String>{'id': 'normals', 'label': 'Normals'},
          <String, String>{'id': 'matcap', 'label': 'Matcap'},
          <String, String>{'id': 'step_heatmap', 'label': 'Step Heatmap'},
          <String, String>{'id': 'cross_section', 'label': 'Cross-Section'},
        ],
        'shading_mode_id': 'full',
        'shading_mode_label': 'Full',
        'show_grid': true,
        'shadows_enabled': false,
        'shadow_steps': 32,
        'ao_enabled': true,
        'ao_samples': 5,
        'ao_intensity': 3.0,
        'march_max_steps': 128,
        'sculpt_fast_mode': false,
        'auto_reduce_steps': true,
        'interaction_render_scale': 0.5,
        'rest_render_scale': 1.0,
        'fog_enabled': false,
        'fog_density': 0.02,
        'bloom_enabled': false,
        'bloom_intensity': 0.3,
        'gamma': 2.2,
        'tonemapping_aces': false,
        'cross_section_axis': 0,
        'cross_section_position': 0.0,
      };

  static final String _renderSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{'render': _defaultRenderPayload()},
  );

  static Map<String, Object?> _defaultSettingsPayload() => <String, Object?>{
        'show_fps_overlay': true,
        'show_node_labels': false,
        'show_bounding_box': true,
        'show_light_gizmos': true,
        'auto_save_enabled': true,
        'auto_save_interval_secs': 120,
        'max_export_resolution': 2048,
        'max_sculpt_resolution': 320,
        'camera_bookmarks': List<Map<String, Object?>>.generate(
          9,
          (index) => <String, Object?>{
            'slot_index': index,
            'saved': index == 0,
          },
          growable: false,
        ),
        'key_options': <Map<String, String>>[
          <String, String>{'id': 'z', 'label': 'Z'},
          <String, String>{'id': 'y', 'label': 'Y'},
          <String, String>{'id': 'u', 'label': 'U'},
          <String, String>{'id': 'f1', 'label': 'F1'},
        ],
        'keybindings': <Map<String, Object?>>[
          <String, Object?>{
            'action_id': 'undo',
            'action_label': 'Undo',
            'category': 'General',
            'binding': <String, Object?>{
              'key_id': 'z',
              'key_label': 'Z',
              'ctrl': true,
              'shift': false,
              'alt': false,
              'shortcut_label': 'Ctrl+Z',
            },
          },
          <String, Object?>{
            'action_id': 'redo',
            'action_label': 'Redo',
            'category': 'General',
            'binding': <String, Object?>{
              'key_id': 'y',
              'key_label': 'Y',
              'ctrl': true,
              'shift': false,
              'alt': false,
              'shortcut_label': 'Ctrl+Y',
            },
          },
          <String, Object?>{
            'action_id': 'toggle_help',
            'action_label': 'Toggle Help',
            'category': 'General',
            'binding': <String, Object?>{
              'key_id': 'f1',
              'key_label': 'F1',
              'ctrl': false,
              'shift': false,
              'alt': false,
              'shortcut_label': 'F1',
            },
          },
        ],
      };

  static Map<String, Object?> _settingsWithKeybindings(
    List<Map<String, Object?>> keybindings,
  ) {
    final settings = Map<String, Object?>.from(_defaultSettingsPayload());
    final defaultKeyOptions =
        (settings['key_options'] as List<dynamic>? ?? const <dynamic>[])
            .map((option) => Map<String, String>.from(option as Map))
            .toList(growable: false);
    final keyOptionsById = <String, Map<String, String>>{
      for (final option in defaultKeyOptions) option['id']!: option,
    };
    for (final keybinding in keybindings) {
      final binding = keybinding['binding'] as Map<String, Object?>?;
      if (binding == null) {
        continue;
      }
      final keyId = binding['key_id'] as String;
      final keyLabel = binding['key_label'] as String;
      keyOptionsById[keyId] = <String, String>{'id': keyId, 'label': keyLabel};
    }
    settings['key_options'] = keyOptionsById.values.toList(growable: false);
    settings['keybindings'] = keybindings;
    return settings;
  }

  static Map<String, Object?> _keybinding({
    required String actionId,
    required String actionLabel,
    required String keyId,
    bool ctrl = false,
    bool shift = false,
    bool alt = false,
    String category = 'General',
  }) {
    final keyLabel = _keyLabelForId(keyId);
    return <String, Object?>{
      'action_id': actionId,
      'action_label': actionLabel,
      'category': category,
      'binding': <String, Object?>{
        'key_id': keyId,
        'key_label': keyLabel,
        'ctrl': ctrl,
        'shift': shift,
        'alt': alt,
        'shortcut_label': _shortcutLabelForBinding(
          keyLabel,
          ctrl: ctrl,
          shift: shift,
          alt: alt,
        ),
      },
    };
  }

  static String _keyLabelForId(String keyId) {
    switch (keyId) {
      case 'space':
        return 'Space';
      case 'enter':
        return 'Enter';
      case 'escape':
        return 'Escape';
      case 'tab':
        return 'Tab';
      case 'delete':
        return 'Delete';
      case 'home':
        return 'Home';
      case 'end':
        return 'End';
      case 'arrow_up':
        return 'Arrow Up';
      case 'arrow_down':
        return 'Arrow Down';
      case 'arrow_left':
        return 'Arrow Left';
      case 'arrow_right':
        return 'Arrow Right';
      case 'open_bracket':
        return '[';
      case 'close_bracket':
        return ']';
      case 'slash':
        return '/';
      default:
        return keyId.toUpperCase();
    }
  }

  static String _shortcutLabelForBinding(
    String keyLabel, {
    required bool ctrl,
    required bool shift,
    required bool alt,
  }) {
    return <String>[
      if (ctrl) 'Ctrl',
      if (shift) 'Shift',
      if (alt) 'Alt',
      keyLabel,
    ].join('+');
  }

  static final String _settingsSnapshot = _withFields(
    _baseSnapshot,
    <String, Object?>{'settings': _defaultSettingsPayload()},
  );

  String currentSnapshot = _baseSnapshot;
  int sceneSnapshotCalls = 0;
  int workflowStatusCalls = 0;
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
  int clearSelectedLightCookieCalls = 0;
  int setNodeLightLinkEnabledCalls = 0;
  int setNodeLightMaskCalls = 0;
  int setSelectedLightArrayColorVariationCalls = 0;
  int setSelectedLightArrayCountCalls = 0;
  int setSelectedLightArrayPatternCalls = 0;
  int setSelectedLightArrayRadiusCalls = 0;
  int setSelectedLightCastShadowsCalls = 0;
  int setSelectedLightColorCalls = 0;
  int setSelectedLightColorHueExpressionCalls = 0;
  int setSelectedLightCookieCalls = 0;
  int setSelectedLightIntensityCalls = 0;
  int setSelectedLightIntensityExpressionCalls = 0;
  int setSelectedLightProximityModeCalls = 0;
  int setSelectedLightProximityRangeCalls = 0;
  int setSelectedLightRangeCalls = 0;
  int setSelectedLightShadowColorCalls = 0;
  int setSelectedLightShadowSoftnessCalls = 0;
  int setSelectedLightSpotAngleCalls = 0;
  int setSelectedLightTypeCalls = 0;
  int setSelectedLightVolumetricCalls = 0;
  int setSelectedLightVolumetricDensityCalls = 0;
  int duplicateSelectedCalls = 0;
  int newSceneCalls = 0;
  int openSceneCalls = 0;
  int openRecentSceneCalls = 0;
  int saveSceneCalls = 0;
  int saveSceneAsCalls = 0;
  int recoverAutosaveCalls = 0;
  int discardRecoveryCalls = 0;
  int applyRenderPresetCalls = 0;
  int setRenderShadingModeCalls = 0;
  int setRenderToggleCalls = 0;
  int setRenderIntegerCalls = 0;
  int setRenderScalarCalls = 0;
  int resetSettingsCalls = 0;
  int exportSettingsCalls = 0;
  int importSettingsCalls = 0;
  int setSettingsToggleCalls = 0;
  int setSettingsIntegerCalls = 0;
  int saveCameraBookmarkCalls = 0;
  int restoreCameraBookmarkCalls = 0;
  int clearCameraBookmarkCalls = 0;
  int resetKeymapCalls = 0;
  int exportKeymapCalls = 0;
  int importKeymapCalls = 0;
  int clearKeybindingCalls = 0;
  int setKeybindingCalls = 0;
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
  int beginInteractiveEditCalls = 0;
  int previewSelectedPrimitiveParameterCalls = 0;
  int previewSelectedMaterialFloatCalls = 0;
  int previewSelectedMaterialColorCalls = 0;
  int previewSelectedTransformPositionCalls = 0;
  int previewSelectedTransformRotationDegreesCalls = 0;
  int previewSelectedTransformScaleCalls = 0;
  int previewSelectedTransformCalls = 0;
  int renameNodeCalls = 0;
  int setSelectedPrimitiveParameterCalls = 0;
  int setSelectedMaterialFloatCalls = 0;
  int setSelectedMaterialColorCalls = 0;
  int setSelectedTransformPositionCalls = 0;
  int setSelectedTransformRotationDegreesCalls = 0;
  int setSelectedTransformScaleCalls = 0;
  int setSelectedTransformCalls = 0;
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
    sceneSnapshotCalls = 0;
    workflowStatusCalls = 0;
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
    clearSelectedLightCookieCalls = 0;
    setNodeLightLinkEnabledCalls = 0;
    setNodeLightMaskCalls = 0;
    setSelectedLightArrayColorVariationCalls = 0;
    setSelectedLightArrayCountCalls = 0;
    setSelectedLightArrayPatternCalls = 0;
    setSelectedLightArrayRadiusCalls = 0;
    setSelectedLightCastShadowsCalls = 0;
    setSelectedLightColorCalls = 0;
    setSelectedLightColorHueExpressionCalls = 0;
    setSelectedLightCookieCalls = 0;
    setSelectedLightIntensityCalls = 0;
    setSelectedLightIntensityExpressionCalls = 0;
    setSelectedLightProximityModeCalls = 0;
    setSelectedLightProximityRangeCalls = 0;
    setSelectedLightRangeCalls = 0;
    setSelectedLightShadowColorCalls = 0;
    setSelectedLightShadowSoftnessCalls = 0;
    setSelectedLightSpotAngleCalls = 0;
    setSelectedLightTypeCalls = 0;
    setSelectedLightVolumetricCalls = 0;
    setSelectedLightVolumetricDensityCalls = 0;
    duplicateSelectedCalls = 0;
    newSceneCalls = 0;
    openSceneCalls = 0;
    openRecentSceneCalls = 0;
    saveSceneCalls = 0;
    saveSceneAsCalls = 0;
    recoverAutosaveCalls = 0;
    discardRecoveryCalls = 0;
    applyRenderPresetCalls = 0;
    setRenderShadingModeCalls = 0;
    setRenderToggleCalls = 0;
    setRenderIntegerCalls = 0;
    setRenderScalarCalls = 0;
    resetSettingsCalls = 0;
    exportSettingsCalls = 0;
    importSettingsCalls = 0;
    setSettingsToggleCalls = 0;
    setSettingsIntegerCalls = 0;
    saveCameraBookmarkCalls = 0;
    restoreCameraBookmarkCalls = 0;
    clearCameraBookmarkCalls = 0;
    resetKeymapCalls = 0;
    exportKeymapCalls = 0;
    importKeymapCalls = 0;
    clearKeybindingCalls = 0;
    setKeybindingCalls = 0;
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
    beginInteractiveEditCalls = 0;
    previewSelectedPrimitiveParameterCalls = 0;
    previewSelectedMaterialFloatCalls = 0;
    previewSelectedMaterialColorCalls = 0;
    previewSelectedTransformPositionCalls = 0;
    previewSelectedTransformRotationDegreesCalls = 0;
    previewSelectedTransformScaleCalls = 0;
    previewSelectedTransformCalls = 0;
    renameNodeCalls = 0;
    setSelectedPrimitiveParameterCalls = 0;
    setSelectedMaterialFloatCalls = 0;
    setSelectedMaterialColorCalls = 0;
    setSelectedTransformPositionCalls = 0;
    setSelectedTransformRotationDegreesCalls = 0;
    setSelectedTransformScaleCalls = 0;
    setSelectedTransformCalls = 0;
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

  AppSceneSnapshot _currentSceneSnapshot() =>
      parseSceneSnapshotJson(currentSnapshot);

  AppSceneSnapshot _setCurrentSceneSnapshot(String snapshot) {
    currentSnapshot = snapshot;
    return _currentSceneSnapshot();
  }

  AppSceneSnapshot _updateRenderSnapshot(
    void Function(Map<String, dynamic> render) apply,
  ) {
    final snapshot = jsonDecode(currentSnapshot) as Map<String, dynamic>;
    final render = Map<String, dynamic>.from(
      snapshot['render'] as Map<String, dynamic>? ?? _defaultRenderPayload(),
    );
    apply(render);
    snapshot['render'] = render;
    final tool = Map<String, dynamic>.from(snapshot['tool'] as Map<String, dynamic>? ?? <String, Object?>{
      'active_tool_label': 'Select',
      'shading_mode_label': 'Full',
      'grid_enabled': true,
    });
    tool['shading_mode_label'] = render['shading_mode_label'];
    tool['grid_enabled'] = render['show_grid'];
    snapshot['tool'] = tool;
    currentSnapshot = jsonEncode(snapshot);
    return _currentSceneSnapshot();
  }

  AppSceneSnapshot _updateSettingsSnapshot(
    void Function(Map<String, dynamic> settings) apply,
  ) {
    final snapshot = jsonDecode(currentSnapshot) as Map<String, dynamic>;
    final settings = jsonDecode(
          jsonEncode(
            snapshot['settings'] as Map<String, dynamic>? ??
                _defaultSettingsPayload(),
          ),
        )
        as Map<String, dynamic>;
    apply(settings);
    snapshot['settings'] = settings;
    currentSnapshot = jsonEncode(snapshot);
    return _currentSceneSnapshot();
  }

  static Map<String, Object?> _vec3Payload(AppVec3 value) => <String, Object?>{
        'x': value.x,
        'y': value.y,
        'z': value.z,
      };

  AppSceneSnapshot _updateTransformSnapshot({
    required AppVec3 position,
    required AppVec3 rotationDegrees,
    AppVec3? scale,
  }) {
    final snapshot = jsonDecode(currentSnapshot) as Map<String, dynamic>;
    final selectedNodeProperties = Map<String, dynamic>.from(
      snapshot['selected_node_properties'] as Map<String, dynamic>? ??
          <String, dynamic>{},
    );
    final transform = Map<String, dynamic>.from(
      selectedNodeProperties['transform'] as Map<String, dynamic>? ??
          <String, dynamic>{
            'position_label': 'Translation',
            'position': _vec3Payload(const AppVec3(x: 0, y: 0, z: 0)),
            'rotation_degrees': _vec3Payload(const AppVec3(x: 0, y: 0, z: 0)),
          },
    );

    transform['position'] = _vec3Payload(position);
    transform['rotation_degrees'] = _vec3Payload(rotationDegrees);
    if (scale != null) {
      transform['scale'] = _vec3Payload(scale);
    }

    selectedNodeProperties['transform'] = transform;
    snapshot['selected_node_properties'] = selectedNodeProperties;
    currentSnapshot = jsonEncode(snapshot);
    return _currentSceneSnapshot();
  }

  static String _renderModeLabel(String modeId) {
    return switch (modeId) {
      'solid' => 'Solid',
      'clay' => 'Clay',
      'normals' => 'Normals',
      'matcap' => 'Matcap',
      'step_heatmap' => 'Step Heatmap',
      'cross_section' => 'Cross-Section',
      _ => 'Full',
    };
  }

  static void _applyRenderPresetToPayload(
    Map<String, dynamic> render,
    String presetId,
  ) {
    switch (presetId) {
      case 'fast':
        render['shadows_enabled'] = false;
        render['ao_enabled'] = false;
        render['march_max_steps'] = 64;
        render['fog_enabled'] = false;
        render['tonemapping_aces'] = false;
        render['sculpt_fast_mode'] = true;
        render['auto_reduce_steps'] = true;
        render['interaction_render_scale'] = 0.35;
        render['rest_render_scale'] = 0.75;
        break;
      case 'quality':
        render['shadows_enabled'] = true;
        render['shadow_steps'] = 64;
        render['ao_enabled'] = true;
        render['ao_samples'] = 8;
        render['ao_intensity'] = 4.0;
        render['march_max_steps'] = 256;
        render['tonemapping_aces'] = true;
        render['sculpt_fast_mode'] = false;
        render['auto_reduce_steps'] = false;
        render['interaction_render_scale'] = 0.5;
        render['rest_render_scale'] = 1.0;
        break;
      default:
        render['shadows_enabled'] = false;
        render['ao_enabled'] = true;
        render['ao_samples'] = 5;
        render['ao_intensity'] = 3.0;
        render['march_max_steps'] = 128;
        render['sculpt_fast_mode'] = false;
        render['auto_reduce_steps'] = true;
        render['interaction_render_scale'] = 0.5;
        render['rest_render_scale'] = 1.0;
        break;
    }
  }

  @override
  AppSceneSnapshot crateApiSimpleAddBox() {
    addBoxCalls += 1;
    currentSnapshot = _selectedUndoSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleAddCylinder() => _setCurrentSceneSnapshot(_selectedUndoSnapshot);

  @override
  AppSceneSnapshot crateApiSimpleAddSphere() {
    currentSnapshot = _selectedUndoSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleAddTorus() => _setCurrentSceneSnapshot(_selectedUndoSnapshot);

  @override
  String crateApiSimpleBridgeVersion() => '0.1.0-test';

  @override
  AppSceneSnapshot crateApiSimpleDiscardRecovery() {
    discardRecoveryCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleApplyRenderPreset({required String presetId}) {
    applyRenderPresetCalls += 1;
    return _updateRenderSnapshot(
      (render) => _applyRenderPresetToPayload(render, presetId),
    );
  }

  @override
  AppSceneSnapshot crateApiSimpleSetRenderShadingMode({required String modeId}) {
    setRenderShadingModeCalls += 1;
    return _updateRenderSnapshot((render) {
      render['shading_mode_id'] = modeId;
      render['shading_mode_label'] = _renderModeLabel(modeId);
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetRenderToggle({
    required String fieldId,
    required bool enabled,
  }) {
    setRenderToggleCalls += 1;
    return _updateRenderSnapshot((render) {
      render[fieldId] = enabled;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetRenderInteger({
    required String fieldId,
    required int value,
  }) {
    setRenderIntegerCalls += 1;
    return _updateRenderSnapshot((render) {
      switch (fieldId) {
        case 'shadow_steps':
          render[fieldId] = value.clamp(8, 128);
        case 'ao_samples':
          render[fieldId] = value.clamp(1, 16);
        case 'march_max_steps':
          render[fieldId] = value.clamp(32, 512);
        case 'cross_section_axis':
          render[fieldId] = value.clamp(0, 2);
        default:
          render[fieldId] = value;
      }
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetRenderScalar({
    required String fieldId,
    required double value,
  }) {
    setRenderScalarCalls += 1;
    return _updateRenderSnapshot((render) {
      switch (fieldId) {
        case 'ao_intensity':
          render[fieldId] = value.clamp(0.5, 10.0);
        case 'fog_density':
          render[fieldId] = value.clamp(0.001, 0.2);
        case 'bloom_intensity':
          render[fieldId] = value.clamp(0.05, 2.0);
        case 'gamma':
          render[fieldId] = value.clamp(1.0, 3.0);
        case 'interaction_render_scale':
          render[fieldId] = value.clamp(0.25, 1.0);
        case 'rest_render_scale':
          render[fieldId] = value.clamp(0.5, 1.0);
        case 'cross_section_position':
          render[fieldId] = value.clamp(-5.0, 5.0);
        default:
          render[fieldId] = value;
      }
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleResetSettings() {
    resetSettingsCalls += 1;
    return _updateSettingsSnapshot((settings) {
      settings
        ..clear()
        ..addAll(_defaultSettingsPayload());
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleExportSettings() {
    exportSettingsCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleImportSettings() {
    importSettingsCalls += 1;
    return _updateSettingsSnapshot((settings) {
      settings
        ..clear()
        ..addAll(_defaultSettingsPayload());
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSettingsToggle({
    required String fieldId,
    required bool enabled,
  }) {
    setSettingsToggleCalls += 1;
    return _updateSettingsSnapshot((settings) {
      settings[fieldId] = enabled;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSettingsInteger({
    required String fieldId,
    required int value,
  }) {
    setSettingsIntegerCalls += 1;
    return _updateSettingsSnapshot((settings) {
      switch (fieldId) {
        case 'auto_save_interval_secs':
          settings[fieldId] = value.clamp(30, 600);
        case 'max_export_resolution':
          settings[fieldId] = value.clamp(64, 4096);
        case 'max_sculpt_resolution':
          settings[fieldId] = value.clamp(16, 512);
        default:
          settings[fieldId] = value;
      }
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSaveCameraBookmark({required int slotIndex}) {
    saveCameraBookmarkCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final bookmarks = List<dynamic>.from(
        settings['camera_bookmarks'] as List<dynamic>? ?? const <dynamic>[],
      );
      if (slotIndex < 0 || slotIndex >= bookmarks.length) {
        return;
      }
      bookmarks[slotIndex] = <String, Object?>{
        'slot_index': slotIndex,
        'saved': true,
      };
      settings['camera_bookmarks'] = bookmarks;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleRestoreCameraBookmark({required int slotIndex}) {
    restoreCameraBookmarkCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleClearCameraBookmark({required int slotIndex}) {
    clearCameraBookmarkCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final bookmarks = List<dynamic>.from(
        settings['camera_bookmarks'] as List<dynamic>? ?? const <dynamic>[],
      );
      if (slotIndex < 0 || slotIndex >= bookmarks.length) {
        return;
      }
      bookmarks[slotIndex] = <String, Object?>{
        'slot_index': slotIndex,
        'saved': false,
      };
      settings['camera_bookmarks'] = bookmarks;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleResetKeymap() {
    resetKeymapCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final defaults = _defaultSettingsPayload();
      settings['key_options'] = defaults['key_options'];
      settings['keybindings'] = defaults['keybindings'];
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleExportKeymap() {
    exportKeymapCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleImportKeymap() {
    importKeymapCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final defaults = _defaultSettingsPayload();
      settings['key_options'] = defaults['key_options'];
      settings['keybindings'] = defaults['keybindings'];
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleClearKeybinding({required String actionId}) {
    clearKeybindingCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final keybindings = List<Map<String, dynamic>>.from(
        (settings['keybindings'] as List<dynamic>? ?? const <dynamic>[])
            .map((binding) => Map<String, dynamic>.from(binding as Map)),
      );
      final bindingIndex = keybindings.indexWhere(
        (binding) => binding['action_id'] == actionId,
      );
      if (bindingIndex == -1) {
        return;
      }
      keybindings[bindingIndex]['binding'] = null;
      settings['keybindings'] = keybindings;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetKeybinding({
    required String actionId,
    required String keyId,
    required bool ctrl,
    required bool shift,
    required bool alt,
  }) {
    setKeybindingCalls += 1;
    return _updateSettingsSnapshot((settings) {
      final keybindings = List<Map<String, dynamic>>.from(
        (settings['keybindings'] as List<dynamic>? ?? const <dynamic>[])
            .map((binding) => Map<String, dynamic>.from(binding as Map)),
      );
      final shortcutLabelParts = <String>[
        if (ctrl) 'Ctrl',
        if (shift) 'Shift',
        if (alt) 'Alt',
        keyId.toUpperCase(),
      ];
      final nextBinding = <String, Object?>{
        'key_id': keyId,
        'key_label': keyId.toUpperCase(),
        'ctrl': ctrl,
        'shift': shift,
        'alt': alt,
        'shortcut_label': shortcutLabelParts.join('+'),
      };
      for (final binding in keybindings) {
        final existing = binding['binding'] as Map<String, dynamic>?;
        if (binding['action_id'] != actionId &&
            existing != null &&
            existing['key_id'] == keyId &&
            existing['ctrl'] == ctrl &&
            existing['shift'] == shift &&
            existing['alt'] == alt) {
          binding['binding'] = null;
        }
      }
      final bindingIndex = keybindings.indexWhere(
        (binding) => binding['action_id'] == actionId,
      );
      if (bindingIndex == -1) {
        return;
      }
      keybindings[bindingIndex]['binding'] = nextBinding;
      settings['keybindings'] = keybindings;
    });
  }

  @override
  AppSceneSnapshot crateApiSimpleSetExportResolution({required int resolution}) {
    setExportResolutionCalls += 1;
    currentSnapshot = _exportResolutionSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetAdaptiveExport({required bool enabled}) {
    setAdaptiveExportCalls += 1;
    currentSnapshot = enabled ? _exportAdaptiveSnapshot : _exportResolutionSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleStartExport() {
    startExportCalls += 1;
    currentSnapshot = _exportRunningSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCancelExport() {
    cancelExportCalls += 1;
    currentSnapshot = _exportIdleSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleOpenImportDialog() {
    openImportDialogCalls += 1;
    currentSnapshot = _importDialogSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCancelImportDialog() {
    cancelImportDialogCalls += 1;
    currentSnapshot = _baseSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetImportUseAuto({required bool useAuto}) {
    setImportUseAutoCalls += 1;
    currentSnapshot = useAuto ? _importDialogSnapshot : _importManualSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetImportResolution({required int resolution}) {
    setImportResolutionCalls += 1;
    currentSnapshot = _importResolutionSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleStartImport() {
    startImportCalls += 1;
    currentSnapshot = _importRunningSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCancelImport() {
    cancelImportCalls += 1;
    currentSnapshot = _baseSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleOpenSculptConvertDialogForSelected() {
    openSculptConvertDialogCalls += 1;
    currentSnapshot = _sculptConvertDialogSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCancelSculptConvertDialog() {
    cancelSculptConvertDialogCalls += 1;
    currentSnapshot = _selectedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptConvertMode({required String modeId}) {
    setSculptConvertModeCalls += 1;
    currentSnapshot = modeId == 'whole_scene_flatten'
        ? _sculptConvertFlattenSnapshot
        : _sculptConvertDialogSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptConvertResolution({required int resolution}) {
    setSculptConvertResolutionCalls += 1;
    currentSnapshot = _sculptConvertResolutionSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleStartSculptConvert() {
    startSculptConvertCalls += 1;
    currentSnapshot = _sculptConvertRunningSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCameraBack() => _currentSceneSnapshot();

  @override
  AppSceneSnapshot crateApiSimpleCameraBottom() => _currentSceneSnapshot();

  @override
  void crateApiSimpleBeginInteractiveEdit() {
    beginInteractiveEditCalls += 1;
  }

  @override
  AppSceneSnapshot crateApiSimpleCameraFront() {
    cameraFrontCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCameraLeft() => _currentSceneSnapshot();

  @override
  AppSceneSnapshot crateApiSimpleCameraRight() => _currentSceneSnapshot();

  @override
  AppSceneSnapshot crateApiSimpleCameraTop() => _currentSceneSnapshot();

  @override
  AppSceneSnapshot crateApiSimpleCreateModifier({required String modifierId}) {
    createModifierCalls += 1;
    currentSnapshot = _modifierSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCreateLight({required String lightId}) {
    createLightCalls += 1;
    currentSnapshot = _lightSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCreateOperation({required String operationId}) {
    createOperationCalls += 1;
    currentSnapshot = _operationSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCreateTransform() {
    createTransformCalls += 1;
    currentSnapshot = _transformSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleCreateSculpt() {
    createSculptCalls += 1;
    currentSnapshot = _sculptActiveSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleResumeSculptingSelected() {
    resumeSculptingSelectedCalls += 1;
    currentSnapshot = _sculptActiveSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleStopSculpting() {
    stopSculptingCalls += 1;
    currentSnapshot = _sculptStoppedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptBrushMode({required String modeId}) {
    setSculptBrushModeCalls += 1;
    currentSnapshot = modeId == 'grab' ? _sculptGrabSnapshot : _sculptActiveSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptBrushRadius({required double radius}) {
    setSculptBrushRadiusCalls += 1;
    currentSnapshot = _sculptActiveSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptBrushStrength({required double strength}) {
    setSculptBrushStrengthCalls += 1;
    currentSnapshot = _sculptGrabSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSculptSymmetryAxis({required String axisId}) {
    setSculptSymmetryAxisCalls += 1;
    currentSnapshot = axisId == 'z' ? _sculptSymmetrySnapshot : _sculptActiveSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedSculptResolution({required int resolution}) {
    setSelectedSculptResolutionCalls += 1;
    currentSnapshot = _sculptResolutionSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleDeleteSelected() {
    deleteSelectedCalls += 1;
    currentSnapshot = _baseUndoSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleNewScene() {
    newSceneCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleOpenRecentScene({required String path}) {
    openRecentSceneCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleOpenScene() {
    openSceneCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleDuplicateSelected() {
    duplicateSelectedCalls += 1;
    currentSnapshot = _duplicatedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleRenameNode({
    required BigInt nodeId,
    required String name,
  }) {
    renameNodeCalls += 1;
    currentSnapshot = _renamedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedPrimitiveParameter({
    required String parameterKey,
    required double value,
  }) {
    setSelectedPrimitiveParameterCalls += 1;
    currentSnapshot = _selectedPropertyRadiusSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedPrimitiveParameter({
    required String parameterKey,
    required double value,
  }) {
    previewSelectedPrimitiveParameterCalls += 1;
    currentSnapshot = _selectedPropertyRadiusSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedMaterialFloat({
    required String fieldId,
    required double value,
  }) {
    setSelectedMaterialFloatCalls += 1;
    currentSnapshot = _selectedPropertyRoughnessSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedMaterialFloat({
    required String fieldId,
    required double value,
  }) {
    previewSelectedMaterialFloatCalls += 1;
    currentSnapshot = _selectedPropertyRoughnessSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedMaterialColor({
    required String fieldId,
    required double red,
    required double green,
    required double blue,
  }) {
    setSelectedMaterialColorCalls += 1;
    currentSnapshot = _selectedPropertyColorSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedMaterialColor({
    required String fieldId,
    required double red,
    required double green,
    required double blue,
  }) {
    previewSelectedMaterialColorCalls += 1;
    currentSnapshot = _selectedPropertyColorSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedTransformPosition({
    required double x,
    required double y,
    required double z,
  }) {
    setSelectedTransformPositionCalls += 1;
    currentSnapshot = _selectedTransformMovedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedTransformPosition({
    required double x,
    required double y,
    required double z,
  }) {
    previewSelectedTransformPositionCalls += 1;
    currentSnapshot = _selectedTransformMovedSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedTransformRotationDegrees({
    required double xDegrees,
    required double yDegrees,
    required double zDegrees,
  }) {
    setSelectedTransformRotationDegreesCalls += 1;
    currentSnapshot = _selectedTransformRotatedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedTransformRotationDegrees({
    required double xDegrees,
    required double yDegrees,
    required double zDegrees,
  }) {
    previewSelectedTransformRotationDegreesCalls += 1;
    currentSnapshot = _selectedTransformRotatedSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedTransformScale({
    required double x,
    required double y,
    required double z,
  }) {
    setSelectedTransformScaleCalls += 1;
    currentSnapshot = _selectedTransformScaledSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  void crateApiSimplePreviewSelectedTransformScale({
    required double x,
    required double y,
    required double z,
  }) {
    previewSelectedTransformScaleCalls += 1;
    currentSnapshot = _selectedTransformScaledSnapshot;
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedTransform({
    required AppVec3 position,
    required AppVec3 rotationDegrees,
    AppVec3? scale,
  }) {
    setSelectedTransformCalls += 1;
    return _updateTransformSnapshot(
      position: position,
      rotationDegrees: rotationDegrees,
      scale: scale,
    );
  }

  @override
  void crateApiSimplePreviewSelectedTransform({
    required AppVec3 position,
    required AppVec3 rotationDegrees,
    AppVec3? scale,
  }) {
    previewSelectedTransformCalls += 1;
    _updateTransformSnapshot(
      position: position,
      rotationDegrees: rotationDegrees,
      scale: scale,
    );
  }

  @override
  AppSceneSnapshot crateApiSimpleClearSelectedLightCookie() {
    clearSelectedLightCookieCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetNodeLightLinkEnabled({
    required BigInt nodeId,
    required BigInt lightId,
    required bool enabled,
  }) {
    setNodeLightLinkEnabledCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetNodeLightMask({
    required BigInt nodeId,
    required int mask,
  }) {
    setNodeLightMaskCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightArrayColorVariation({
    required double value,
  }) {
    setSelectedLightArrayColorVariationCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightArrayCount({required int count}) {
    setSelectedLightArrayCountCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightArrayPattern({
    required String patternId,
  }) {
    setSelectedLightArrayPatternCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightArrayRadius({required double radius}) {
    setSelectedLightArrayRadiusCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightCastShadows({required bool enabled}) {
    setSelectedLightCastShadowsCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightColor({
    required double red,
    required double green,
    required double blue,
  }) {
    setSelectedLightColorCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightColorHueExpression({
    required String expression,
  }) {
    setSelectedLightColorHueExpressionCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightCookie({required BigInt cookieNodeId}) {
    setSelectedLightCookieCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightIntensity({required double intensity}) {
    setSelectedLightIntensityCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightIntensityExpression({
    required String expression,
  }) {
    setSelectedLightIntensityExpressionCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightProximityMode({required String modeId}) {
    setSelectedLightProximityModeCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightProximityRange({required double range}) {
    setSelectedLightProximityRangeCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightRange({required double range}) {
    setSelectedLightRangeCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightShadowColor({
    required double red,
    required double green,
    required double blue,
  }) {
    setSelectedLightShadowColorCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightShadowSoftness({
    required double softness,
  }) {
    setSelectedLightShadowSoftnessCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightSpotAngle({
    required double angleDegrees,
  }) {
    setSelectedLightSpotAngleCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightType({required String lightTypeId}) {
    setSelectedLightTypeCalls += 1;
    currentSnapshot = lightTypeId == 'array'
        ? _arrayLightSnapshot
        : _advancedLightSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightVolumetric({required bool enabled}) {
    setSelectedLightVolumetricCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetSelectedLightVolumetricDensity({
    required double density,
  }) {
    setSelectedLightVolumetricDensityCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSetManipulatorMode({required String modeId}) {
    setManipulatorModeCalls += 1;
    currentSnapshot = switch (modeId) {
      'rotate' => _selectedTransformRotateSnapshot,
      _ => _selectedTransformManipulatorSnapshot,
    };
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleToggleManipulatorSpace() {
    toggleManipulatorSpaceCalls += 1;
    currentSnapshot = _selectedTransformRotateWorldSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleNudgeManipulatorPivotOffset({
    required double x,
    required double y,
    required double z,
  }) {
    nudgeManipulatorPivotOffsetCalls += 1;
    currentSnapshot = _selectedTransformPivotSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleResetManipulatorPivot() {
    resetManipulatorPivotCalls += 1;
    currentSnapshot = _selectedTransformPivotResetSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleNudgeSelectedTranslation({
    required double deltaX,
    required double deltaY,
    required double deltaZ,
  }) {
    nudgeSelectedTranslationCalls += 1;
    currentSnapshot = _selectedTransformMovedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleNudgeSelectedRotationDegrees({
    required double deltaXDegrees,
    required double deltaYDegrees,
    required double deltaZDegrees,
  }) {
    nudgeSelectedRotationDegreesCalls += 1;
    currentSnapshot = _selectedTransformRotatedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleNudgeSelectedScale({
    required double deltaX,
    required double deltaY,
    required double deltaZ,
  }) {
    nudgeSelectedScaleCalls += 1;
    currentSnapshot = _selectedTransformScaledSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleFocusSelected() {
    focusSelectedCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleFrameAll() => _currentSceneSnapshot();

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
  AppSceneSnapshot crateApiSimpleRecoverAutosave() {
    recoverAutosaveCalls += 1;
    return _currentSceneSnapshot();
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
  AppSceneSnapshot crateApiSimpleResetScene() {
    currentSnapshot = _baseSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSaveScene() {
    saveSceneCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSaveSceneAs() {
    saveSceneAsCalls += 1;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleRedo() {
    redoCalls += 1;
    currentSnapshot = _selectedUndoSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppWorkflowStatusSnapshot crateApiSimpleWorkflowStatus() {
    workflowStatusCalls += 1;
    var sceneChanged = false;

    if (currentSnapshot == _exportRunningSnapshot && workflowStatusCalls >= 3) {
      currentSnapshot = _exportDoneSnapshot;
    } else if (currentSnapshot == _importRunningSnapshot &&
        workflowStatusCalls >= 3) {
      currentSnapshot = _importDoneSnapshot;
      sceneChanged = true;
    } else if (currentSnapshot == _sculptConvertRunningSnapshot &&
        workflowStatusCalls >= 3) {
      currentSnapshot = _sculptConvertDoneSnapshot;
      sceneChanged = true;
    }

    final snapshot = parseSceneSnapshotJson(currentSnapshot);
    return AppWorkflowStatusSnapshot(
      exportStatus: snapshot.export_.status,
      importStatus: snapshot.import_.status,
      sculptConvertStatus: snapshot.sculptConvert.status,
      sceneChanged: sceneChanged,
    );
  }

  @override
  AppSceneSnapshot crateApiSimpleSceneSnapshot() {
    sceneSnapshotCalls += 1;
    return parseSceneSnapshotJson(currentSnapshot);
  }

  @override
  AppSceneSnapshot crateApiSimpleSelectNode({BigInt? nodeId}) {
    selectNodeCalls += 1;
    currentSnapshot = nodeId == null ? _baseSnapshot : _selectedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleSelectNodeAtViewport({
    required double mouseX,
    required double mouseY,
    required int width,
    required int height,
    required double timeSeconds,
  }) {
    currentSnapshot = _selectedSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleToggleNodeLock({required BigInt nodeId}) {
    toggleNodeLockCalls += 1;
    currentSnapshot = switch (currentSnapshot) {
      _selectedPropertySnapshot => _selectedPropertyLockedSnapshot,
      _selectedPropertyLockedSnapshot => _selectedPropertySnapshot,
      _ => _selectedUndoSnapshot,
    };
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleToggleNodeVisibility({required BigInt nodeId}) {
    toggleNodeVisibilityCalls += 1;
    currentSnapshot = switch (currentSnapshot) {
      _selectedPropertySnapshot => _selectedPropertyHiddenSnapshot,
      _selectedPropertyHiddenSnapshot => _selectedPropertySnapshot,
      _ => _selectedUndoSnapshot,
    };
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleToggleOrthographic() {
    toggleOrthographicCalls += 1;
    currentSnapshot = currentSnapshot == _orthoSnapshot
        ? _baseSnapshot
        : _orthoSnapshot;
    return _currentSceneSnapshot();
  }

  @override
  AppSceneSnapshot crateApiSimpleUndo() {
    undoCalls += 1;
    currentSnapshot = _baseRedoSnapshot;
    return _currentSceneSnapshot();
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
      scrollable: _commandPanelScrollable(),
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
    Object? feedback,
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
      ...?feedback == null ? null : <Object?, Object?>{'feedback': feedback},
      ...?hostError == null ? null : <Object?, Object?>{'hostError': hostError},
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

    final movedSnapshot = mockApi.crateApiSimpleSetSelectedTransformPosition(
      x: 1.0,
      y: -2.0,
      z: 3.5,
    );
    final rotatedSnapshot =
        mockApi.crateApiSimpleSetSelectedTransformRotationDegrees(
      xDegrees: 10.0,
      yDegrees: 20.0,
      zDegrees: 30.0,
    );
    final scaledSnapshot = mockApi.crateApiSimpleSetSelectedTransformScale(
      x: -1.0,
      y: 2.0,
      z: 500.0,
    );

    expect(mockApi.setSelectedTransformPositionCalls, 1);
    expect(mockApi.setSelectedTransformRotationDegreesCalls, 1);
    expect(mockApi.setSelectedTransformScaleCalls, 1);
    expect(
      movedSnapshot.selectedNodeProperties?.transform?.position,
      const AppVec3(x: 1.0, y: -2.0, z: 3.5),
    );
    expect(
      rotatedSnapshot.selectedNodeProperties?.transform?.rotationDegrees,
      const AppVec3(x: 10.0, y: 20.0, z: 30.0),
    );
    expect(
      scaledSnapshot.selectedNodeProperties?.transform?.scale,
      const AppVec3(x: 0.01, y: 2.0, z: 100.0),
    );
  });

  testWidgets('routes transform inspector edits through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedTransformPropertySnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-position-x-slider')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-position-x-slider')),
      const Offset(180, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.beginInteractiveEditCalls, greaterThan(0));
    expect(mockApi.previewSelectedTransformPositionCalls, greaterThan(0));
    expect(mockApi.setSelectedTransformPositionCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformMovedSnapshot);

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-rotation-z-slider')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-rotation-z-slider')),
      const Offset(120, 0),
    );
    await tester.pumpAndSettle();

    expect(
      mockApi.previewSelectedTransformRotationDegreesCalls,
      greaterThan(0),
    );
    expect(mockApi.setSelectedTransformRotationDegreesCalls, 1);
    expect(
      mockApi.currentSnapshot,
      _MockRustApi._selectedTransformRotatedSnapshot,
    );

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('transform-scale-x-slider')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('transform-scale-x-slider')),
      const Offset(-200, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.previewSelectedTransformScaleCalls, greaterThan(0));
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

  testWidgets('keyboard shortcut routes gizmo mode through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._withFields(
      _MockRustApi._selectedTransformManipulatorSnapshot,
      <String, Object?>{
        'settings': _MockRustApi._settingsWithKeybindings(<Map<String, Object?>>[
          _MockRustApi._keybinding(
            actionId: 'gizmo_rotate',
            actionLabel: 'Gizmo Rotate',
            keyId: 'e',
            category: 'Viewport',
          ),
        ]),
      },
    );

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyE);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyE);
    await tester.pump();

    expect(mockApi.setManipulatorModeCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedTransformRotateSnapshot);
  });

  testWidgets('keyboard shortcut routes undo through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._withFields(
      _MockRustApi._selectedTransformManipulatorSnapshot,
      <String, Object?>{'settings': _MockRustApi._defaultSettingsPayload()},
    );

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.keyZ);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.keyZ);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();

    expect(mockApi.undoCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._baseRedoSnapshot);
  });

  testWidgets('viewport light billboard follows gizmo preview during drag', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._withFields(
      _MockRustApi._lightBillboardSnapshot,
      <String, Object?>{
        'tool': <String, Object?>{
          'active_tool_label': 'Select',
          'shading_mode_label': 'Full',
          'grid_enabled': true,
          'manipulator_mode_id': 'translate',
          'manipulator_mode_label': 'Move',
          'manipulator_space_id': 'local',
          'manipulator_space_label': 'Local',
          'manipulator_visible': true,
          'can_reset_pivot': false,
          'pivot_offset': <String, double>{'x': 0.0, 'y': 0.0, 'z': 0.0},
        },
      },
    );

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    final dynamic state = tester.state(find.byType(BridgeStatusPage));
    final viewportRect = tester.getRect(find.byType(ViewportSurface));
    var viewportWidth = viewportRect.width;
    var viewportHeight = viewportWidth / (16.0 / 9.0);
    if (viewportHeight > viewportRect.height) {
      viewportHeight = viewportRect.height;
      viewportWidth = viewportHeight * (16.0 / 9.0);
    }
    final innerViewportTopLeft = Offset(
      viewportRect.left + (viewportRect.width - viewportWidth) * 0.5,
      viewportRect.top + (viewportRect.height - viewportHeight) * 0.5,
    );
    final localViewportCenter = Offset(
      viewportWidth * 0.5,
      viewportHeight * 0.5,
    );
    final localBillboardPositionBefore =
        state.debugViewportLightBillboardPosition(nodeId: BigInt.from(8))
            as Offset?;
    final localHandlePosition =
        state.debugViewportGizmoAxisHandlePosition('x') as Offset? ??
        state.debugViewportGizmoAxisHandlePosition('y') as Offset? ??
        state.debugViewportGizmoAxisHandlePosition('z') as Offset?;

    expect(localBillboardPositionBefore, isNotNull);
    expect(localHandlePosition, isNotNull);

    final globalHandlePosition = innerViewportTopLeft + localHandlePosition!;
    final dragDirection =
        globalHandlePosition - (innerViewportTopLeft + localViewportCenter);
    final dragDistance = dragDirection.distance;
    expect(dragDistance, greaterThan(0.0));
    final dragStep = Offset(
      dragDirection.dx / dragDistance * 36.0,
      dragDirection.dy / dragDistance * 36.0,
    );

    final gesture = await tester.startGesture(
      globalHandlePosition,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await tester.pump();
    await gesture.moveBy(dragStep);
    await tester.pump();

    final localBillboardPositionDuring =
        state.debugViewportLightBillboardPosition(nodeId: BigInt.from(8))
            as Offset?;
    expect(localBillboardPositionDuring, isNotNull);
    expect(
      (localBillboardPositionDuring! - localBillboardPositionBefore!).distance,
      greaterThan(0.0),
    );
    expect(mockApi.previewSelectedTransformCalls, greaterThan(0));
    expect(mockApi.setSelectedTransformCalls, 0);

    await gesture.up();
    await tester.pump();

    expect(mockApi.setSelectedTransformCalls, 1);
  });

  testWidgets('viewport gizmo drags preview and commit without orbiting', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._selectedTransformManipulatorSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    final dynamic state = tester.state(find.byType(BridgeStatusPage));
    final viewportRect = tester.getRect(find.byType(ViewportSurface));
    var viewportWidth = viewportRect.width;
    var viewportHeight = viewportWidth / (16.0 / 9.0);
    if (viewportHeight > viewportRect.height) {
      viewportHeight = viewportRect.height;
      viewportWidth = viewportHeight * (16.0 / 9.0);
    }
    final innerViewportTopLeft = Offset(
      viewportRect.left + (viewportRect.width - viewportWidth) * 0.5,
      viewportRect.top + (viewportRect.height - viewportHeight) * 0.5,
    );
    final localViewportCenter = Offset(
      viewportWidth * 0.5,
      viewportHeight * 0.5,
    );
    final localHandlePosition = <Offset?>[
      state.debugViewportGizmoAxisHandlePosition('x') as Offset?,
      state.debugViewportGizmoAxisHandlePosition('y') as Offset?,
      state.debugViewportGizmoAxisHandlePosition('z') as Offset?,
    ]
        .whereType<Offset>()
        .fold<Offset?>(
          null,
          (bestHandle, candidate) {
            if (bestHandle == null) {
              return candidate;
            }
            final candidateDistance =
                (candidate - localViewportCenter).distance;
            final bestDistance = (bestHandle - localViewportCenter).distance;
            return candidateDistance > bestDistance ? candidate : bestHandle;
          },
        );
    expect(localHandlePosition, isNotNull);

    final globalHandlePosition = innerViewportTopLeft + localHandlePosition!;
    final dragDirection =
        globalHandlePosition - (innerViewportTopLeft + localViewportCenter);
    final dragDistance = dragDirection.distance;
    expect(dragDistance, greaterThan(0.0));
    final dragStep = Offset(
      dragDirection.dx / dragDistance * 36.0,
      dragDirection.dy / dragDistance * 36.0,
    );

    final gesture = await tester.startGesture(
      globalHandlePosition,
      kind: PointerDeviceKind.mouse,
      buttons: kPrimaryMouseButton,
    );
    await tester.pump();
    await gesture.moveBy(dragStep);
    await tester.pump();
    await gesture.moveBy(dragStep);
    await tester.pump();
    await gesture.up();
    await tester.pump();

    expect(mockApi.beginInteractiveEditCalls, greaterThan(0));
    expect(mockApi.previewSelectedTransformCalls, greaterThan(0));
    expect(mockApi.setSelectedTransformCalls, 1);
    expect(mockApi.previewSelectedTransformPositionCalls, 0);
    expect(mockApi.setSelectedTransformPositionCalls, 0);
    expect(orbitCalls, 0);

    final transform = parseSceneSnapshotJson(
      mockApi.currentSnapshot,
    ).selectedNodeProperties?.transform;
    expect(transform, isNotNull);
    expect(
      transform!.position.x.abs() +
          transform.position.y.abs() +
          transform.position.z.abs(),
      greaterThan(0.01),
    );
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
      scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pump();

    await tester.tap(find.byKey(const ValueKey('scene-tree-node-1')));
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('scene-tree-visibility-1')));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.text('Box'),
      -200,
      scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();
    await tester.scrollUntilVisible(
      find.text('Radius: 1.00'),
      -120,
      scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('primitive-parameter-radius-slider')),
      const Offset(220, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.beginInteractiveEditCalls, greaterThan(0));
    expect(mockApi.previewSelectedPrimitiveParameterCalls, greaterThan(0));
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('material-roughness-slider')),
      const Offset(180, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.beginInteractiveEditCalls, greaterThan(0));
    expect(mockApi.previewSelectedMaterialFloatCalls, greaterThan(0));
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.drag(
      find.byKey(const ValueKey('material-color-red-slider')),
      const Offset(120, 0),
    );
    await tester.pumpAndSettle();

    expect(mockApi.beginInteractiveEditCalls, greaterThan(0));
    expect(mockApi.previewSelectedMaterialColorCalls, greaterThan(0));
    expect(mockApi.setSelectedMaterialColorCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedPropertyColorSnapshot);
  });

  testWidgets('routes export settings and export start through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._exportIdleSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));
    final requestFrameBaseline = requestFrameCalls;

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('export-resolution-field')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.enterText(
      find.byKey(const ValueKey('export-resolution-field')),
      '256',
    );
    await tester.ensureVisible(find.byKey(const ValueKey('export-apply-resolution')));
    await tester.tap(find.byKey(const ValueKey('export-apply-resolution')));
    await tester.pump();

    await tester.ensureVisible(find.byKey(const ValueKey('export-adaptive-toggle')));
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

    expect(mockApi.workflowStatusCalls, greaterThanOrEqualTo(3));
    expect(mockApi.sceneSnapshotCalls, 1);
    expect(find.text('Exported OBJ (128 verts, 64 tris)'), findsOneWidget);
    expect(find.byKey(const ValueKey('export-progress-indicator')), findsNothing);
    expect(requestFrameCalls, requestFrameBaseline);
  });

  testWidgets('routes render settings controls through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._renderSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));
    final verticalScrollable = find.byWidgetPredicate(
      (widget) => widget is Scrollable && widget.axisDirection == AxisDirection.down,
    ).first;

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('render-preset-quality')),
      200,
      scrollable: verticalScrollable,
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('render-preset-quality')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-shading-cross_section')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-grid-toggle')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('render-shadow-steps-increase')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-shadow-steps-increase')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const ValueKey('render-bloom-toggle')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-bloom-toggle')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('render-bloom-intensity-increase')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-bloom-intensity-increase')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('render-cross-section-axis-z')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('render-cross-section-axis-z')));
    await tester.pumpAndSettle();
    await tester.tap(
      find.byKey(const ValueKey('render-cross-section-position-increase')),
    );
    await tester.pumpAndSettle();

    expect(mockApi.applyRenderPresetCalls, 1);
    expect(mockApi.setRenderShadingModeCalls, 1);
    expect(mockApi.setRenderToggleCalls, 2);
    expect(mockApi.setRenderIntegerCalls, 2);
    expect(mockApi.setRenderScalarCalls, 2);
    expect(mockApi.currentSnapshot, contains('"show_grid":false'));
    expect(mockApi.currentSnapshot, contains('"shadow_steps":72'));
    expect(mockApi.currentSnapshot, contains('"bloom_enabled":true'));
    expect(mockApi.currentSnapshot, contains('"cross_section_axis":2'));
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('routes settings controls through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._settingsSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('settings-reset-command')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('settings-reset-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-export-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-fps-overlay-toggle')));
    await tester.pumpAndSettle();
    await tester.tap(
      find.byKey(const ValueKey('settings-auto-save-interval-increase')),
    );
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('settings-bookmark-save-1')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-bookmark-save-1')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('settings-bookmark-restore-0')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-bookmark-restore-0')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('settings-bookmark-clear-0')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-bookmark-clear-0')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const ValueKey('settings-import-command')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('settings-import-command')));
    await tester.pumpAndSettle();

    expect(mockApi.resetSettingsCalls, 1);
    expect(mockApi.exportSettingsCalls, 1);
    expect(mockApi.importSettingsCalls, 1);
    expect(mockApi.setSettingsToggleCalls, 1);
    expect(mockApi.setSettingsIntegerCalls, 1);
    expect(mockApi.saveCameraBookmarkCalls, 1);
    expect(mockApi.restoreCameraBookmarkCalls, 1);
    expect(mockApi.clearCameraBookmarkCalls, 1);
    expect(mockApi.currentSnapshot, contains('"show_fps_overlay":true'));
    expect(mockApi.currentSnapshot, contains('"auto_save_interval_secs":120'));
    expect(mockApi.currentSnapshot, contains('"slot_index":0,"saved":true'));
    expect(mockApi.currentSnapshot, contains('"slot_index":1,"saved":false'));
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('routes keymap controls through the Rust facade', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._settingsSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('keymap-export-command')),
      200,
      scrollable: _commandPanelScrollable(),
    );
    await tester.pumpAndSettle();

    await tester.tap(find.byKey(const ValueKey('keymap-export-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keymap-import-command')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const ValueKey('keymap-reset-command')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keymap-reset-command')));
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const ValueKey('keybinding-clear-undo')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keybinding-clear-undo')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keybinding-edit-redo')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keybinding-key-dropdown')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keybinding-key-option-z')).last);
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const ValueKey('keybinding-save-command')));
    await tester.pumpAndSettle();

    expect(mockApi.exportKeymapCalls, 1);
    expect(mockApi.importKeymapCalls, 1);
    expect(mockApi.resetKeymapCalls, 1);
    expect(mockApi.clearKeybindingCalls, 1);
    expect(mockApi.setKeybindingCalls, 1);
    expect(mockApi.currentSnapshot, contains('"action_id":"undo"'));
    expect(mockApi.currentSnapshot, contains('"action_id":"redo"'));
    expect(mockApi.currentSnapshot, contains('"key_id":"z"'));
    expect(mockApi.currentSnapshot, contains('"shortcut_label":"Ctrl+Z"'));
    expect(requestFrameCalls, greaterThan(0));
  });

  testWidgets('routes import settings and import start through the Rust facade', (
    WidgetTester tester,
  ) async {
    await pumpApp(tester, logicalSize: const Size(1400, 900));
    final requestFrameBaseline = requestFrameCalls;

    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('open-import-dialog-command')),
      200,
      scrollable: _commandPanelScrollable(),
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

    expect(mockApi.workflowStatusCalls, greaterThanOrEqualTo(3));
    expect(mockApi.sceneSnapshotCalls, 2);
    expect(find.text('Imported hero_mesh.obj as sculpt geometry'), findsOneWidget);
    expect(find.byKey(const ValueKey('import-progress-indicator')), findsNothing);
    expect(requestFrameCalls, greaterThanOrEqualTo(requestFrameBaseline));
  });

  testWidgets(
    'routes sculpt convert settings and start through the Rust facade',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._selectedSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));
      final requestFrameBaseline = requestFrameCalls;

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('open-sculpt-convert-dialog-command')),
        200,
        scrollable: _commandPanelScrollable(),
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

      expect(mockApi.workflowStatusCalls, greaterThanOrEqualTo(3));
      expect(mockApi.sceneSnapshotCalls, 2);
      expect(find.text('Converted Sphere to sculpt'), findsOneWidget);
      expect(
        find.byKey(const ValueKey('sculpt-convert-progress-indicator')),
        findsNothing,
      );
      expect(requestFrameCalls, greaterThanOrEqualTo(requestFrameBaseline));
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
      scrollable: _commandPanelScrollable(),
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
      final verticalScrollable = find.byWidgetPredicate(
        (widget) => widget is Scrollable && widget.axisDirection == AxisDirection.down,
      ).first;

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('rename-command')),
        200,
        scrollable: verticalScrollable,
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
        scrollable: verticalScrollable,
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
        scrollable: _commandPanelScrollable(),
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
    'routes advanced light inspector and linking controls through the Rust facade',
    (WidgetTester tester) async {
      mockApi.currentSnapshot = _MockRustApi._advancedLightSnapshot;

      await pumpApp(tester, logicalSize: const Size(1400, 900));

      await tester.scrollUntilVisible(
        find.byKey(const ValueKey('selected-light-intensity-increase')),
        200,
        scrollable: _commandPanelScrollable(),
      );
      await tester.pumpAndSettle();

      await tester.tap(find.byKey(const ValueKey('selected-light-intensity-increase')));
      await tester.pumpAndSettle();

      await tester.ensureVisible(
        find.byKey(const ValueKey('selected-light-cast-shadows-toggle')),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('selected-light-cast-shadows-toggle')));
      await tester.pumpAndSettle();

      await tester.ensureVisible(
        find.byKey(const ValueKey('selected-light-intensity-expression-field')),
      );
      await tester.pumpAndSettle();
      await tester.enterText(
        find.byKey(const ValueKey('selected-light-intensity-expression-field')),
        'time * 4.0',
      );
      await tester.tap(
        find.byKey(const ValueKey('selected-light-intensity-expression-apply')),
      );
      await tester.pumpAndSettle();

      await tester.ensureVisible(
        find.byKey(const ValueKey('light-link-node-1-light-11')),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('light-link-node-1-light-11')));
      await tester.pumpAndSettle();

      await tester.ensureVisible(
        find.byKey(const ValueKey('selected-light-type-array')),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('selected-light-type-array')));
      await tester.pumpAndSettle();

      await tester.ensureVisible(
        find.byKey(const ValueKey('selected-light-array-pattern-grid')),
      );
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const ValueKey('selected-light-array-pattern-grid')));
      await tester.pumpAndSettle();
      await tester.tap(
        find.byKey(const ValueKey('selected-light-array-count-increase')),
      );
      await tester.pumpAndSettle();

      expect(mockApi.setSelectedLightIntensityCalls, 1);
      expect(mockApi.setSelectedLightCastShadowsCalls, 1);
      expect(mockApi.setSelectedLightIntensityExpressionCalls, 1);
      expect(mockApi.setNodeLightLinkEnabledCalls, 1);
      expect(mockApi.setNodeLightMaskCalls, 0);
      expect(mockApi.setSelectedLightTypeCalls, 1);
      expect(mockApi.setSelectedLightArrayPatternCalls, 1);
      expect(mockApi.setSelectedLightArrayCountCalls, 1);
      expect(mockApi.currentSnapshot, _MockRustApi._arrayLightSnapshot);
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
        scrollable: _commandPanelScrollable(),
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
      scrollable: _commandPanelScrollable(),
    );
    await tester.pump();
    await tester.tap(find.byKey(const ValueKey('undo-command')));
    await tester.pump();
    await tester.scrollUntilVisible(
      find.byKey(const ValueKey('redo-command')),
      200,
      scrollable: _commandPanelScrollable(),
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
    await dispatchTextureEvent(
      tester,
      buildTextureEvent(
        frameCount: 2,
        interactionPhase: 'idle',
      ),
    );

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

      expect(mockApi.sceneSnapshotCalls, 1);

      await dispatchTextureEvent(
        tester,
        buildTextureEvent(
          frameCount: 2,
          interactionPhase: 'interacting',
          feedback: jsonDecode(
            '{"camera":{"yaw":0.9,"pitch":0.5,"roll":0.0,"distance":4.5,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.4,"y":2.1,"z":3.1}},"selected_node":null,"hovered_node":{"id":99,"name":"Hover Probe","kind_label":"Box","visible":true,"locked":false}}',
          ),
        ),
      );

      expect(mockApi.sceneSnapshotCalls, 1);
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
          feedback: jsonDecode(
            '{"camera":{"yaw":0.7853982,"pitch":0.4,"roll":0.0,"distance":5.0,"fov_degrees":45.0,"orthographic":false,"target":{"x":0.0,"y":0.0,"z":0.0},"eye":{"x":3.26,"y":1.95,"z":3.26}},"selected_node":{"id":1,"name":"Sphere","kind_label":"Sphere","visible":true,"locked":false},"hovered_node":null}',
          ),
        ),
      );

      expect(mockApi.sceneSnapshotCalls, 2);
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

  testWidgets('light billboard taps select via Rust without native picking', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._lightBillboardSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    final dynamic state = tester.state(find.byType(BridgeStatusPage));
    final viewportRect = tester.getRect(find.byType(ViewportSurface));
    var viewportWidth = viewportRect.width;
    var viewportHeight = viewportWidth / (16.0 / 9.0);
    if (viewportHeight > viewportRect.height) {
      viewportHeight = viewportRect.height;
      viewportWidth = viewportHeight * (16.0 / 9.0);
    }
    final innerViewportTopLeft = Offset(
      viewportRect.left + (viewportRect.width - viewportWidth) * 0.5,
      viewportRect.top + (viewportRect.height - viewportHeight) * 0.5,
    );
    final localBillboardPosition =
        state.debugViewportLightBillboardPosition(nodeId: BigInt.from(8))
            as Offset?;

    expect(localBillboardPosition, isNotNull);

    await tester.tapAt(innerViewportTopLeft + localBillboardPosition!);
    await tester.pump();

    expect(mockApi.selectNodeCalls, 1);
    expect(mockApi.currentSnapshot, _MockRustApi._selectedSnapshot);
    expect(pickCalls, 0);
  });

  testWidgets('keyboard shortcut saves camera bookmark like egui', (
    WidgetTester tester,
  ) async {
    mockApi.currentSnapshot = _MockRustApi._settingsSnapshot;

    await pumpApp(tester, logicalSize: const Size(1400, 900));

    await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
    await tester.sendKeyDownEvent(LogicalKeyboardKey.digit2);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.digit2);
    await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
    await tester.pump();

    expect(mockApi.saveCameraBookmarkCalls, 1);
    final settings = parseSceneSnapshotJson(mockApi.currentSnapshot).settings;
    expect(settings.cameraBookmarks[1].saved, isTrue);
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
      scrollable: _commandPanelScrollable(),
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



