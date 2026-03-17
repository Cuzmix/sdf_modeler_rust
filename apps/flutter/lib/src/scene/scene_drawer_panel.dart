import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/scene/scene_tree_panel.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_panel_surface.dart';

class SceneDrawerPanel extends StatelessWidget {
  const SceneDrawerPanel({
    super.key,
    required this.roots,
    required this.selectedNodeId,
    required this.selectedNodeIds,
    required this.enabled,
    required this.filterQuery,
    required this.onFilterQueryChanged,
    required this.onSelectNode,
    required this.onToggleNodeSelection,
    required this.onToggleNodeVisibility,
    required this.onToggleNodeLock,
  });

  final List<AppSceneTreeNodeSnapshot> roots;
  final BigInt? selectedNodeId;
  final Set<BigInt> selectedNodeIds;
  final bool enabled;
  final String filterQuery;
  final ValueChanged<String> onFilterQueryChanged;
  final ValueChanged<BigInt> onSelectNode;
  final ValueChanged<BigInt> onToggleNodeSelection;
  final ValueChanged<BigInt> onToggleNodeVisibility;
  final ValueChanged<BigInt> onToggleNodeLock;

  @override
  Widget build(BuildContext context) {
    final normalizedQuery = filterQuery.trim().toLowerCase();
    final filteredRoots = normalizedQuery.isEmpty
        ? roots
        : roots
              .map((node) => _filterNode(node, normalizedQuery))
              .whereType<AppSceneTreeNodeSnapshot>()
              .toList(growable: false);

    return ShellPanelSurface(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Scene Drawer',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ShellTokens.compactGap),
          Text(
            'Search the scene and use ctrl-click or long-press to build a multi-selection.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          TextField(
            key: const ValueKey('scene-drawer-search-field'),
            enabled: enabled,
            decoration: const InputDecoration(
              hintText: 'Filter by name, type, or workflow state',
              prefixIcon: Icon(Icons.search),
            ),
            onChanged: onFilterQueryChanged,
          ),
          const SizedBox(height: ShellTokens.controlGap),
          Expanded(
            child: SingleChildScrollView(
              key: const ValueKey('scene-drawer-scrollable'),
              child: SceneTreePanel(
                roots: filteredRoots,
                selectedNodeId: selectedNodeId,
                selectedNodeIds: selectedNodeIds,
                enabled: enabled,
                onSelectNode: onSelectNode,
                onToggleNodeSelection: onToggleNodeSelection,
                onToggleNodeVisibility: onToggleNodeVisibility,
                onToggleNodeLock: onToggleNodeLock,
              ),
            ),
          ),
        ],
      ),
    );
  }

  AppSceneTreeNodeSnapshot? _filterNode(
    AppSceneTreeNodeSnapshot node,
    String query,
  ) {
    final filteredChildren = node.children
        .map((child) => _filterNode(child, query))
        .whereType<AppSceneTreeNodeSnapshot>()
        .toList(growable: false);
    final matchesNode = node.name.toLowerCase().contains(query) ||
        node.kindLabel.toLowerCase().contains(query) ||
        node.workflowStatusLabel.toLowerCase().contains(query);

    if (!matchesNode && filteredChildren.isEmpty) {
      return null;
    }

    return AppSceneTreeNodeSnapshot(
      id: node.id,
      name: node.name,
      kindLabel: node.kindLabel,
      visible: node.visible,
      locked: node.locked,
      workflowStatusId: node.workflowStatusId,
      workflowStatusLabel: node.workflowStatusLabel,
      children: filteredChildren,
    );
  }
}
