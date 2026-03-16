import 'package:flutter/material.dart';
import 'package:sdf_modeler_flutter/src/rust/api/mirrors.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_contract.dart';
import 'package:sdf_modeler_flutter/src/shell/shell_theme.dart';

class SceneTreePanel extends StatelessWidget {
  const SceneTreePanel({
    super.key,
    required this.roots,
    required this.selectedNodeId,
    required this.enabled,
    required this.onSelectNode,
    required this.onToggleNodeVisibility,
    required this.onToggleNodeLock,
  });

  final List<AppSceneTreeNodeSnapshot> roots;
  final BigInt? selectedNodeId;
  final bool enabled;
  final ValueChanged<BigInt> onSelectNode;
  final ValueChanged<BigInt> onToggleNodeVisibility;
  final ValueChanged<BigInt> onToggleNodeLock;

  @override
  Widget build(BuildContext context) {
    if (roots.isEmpty) {
      return const Text('No nodes in the current scene.');
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: roots
          .map(
            (node) => _SceneTreeNodeTile(
              node: node,
              depth: 0,
              selectedNodeId: selectedNodeId,
              enabled: enabled,
              onSelectNode: onSelectNode,
              onToggleNodeVisibility: onToggleNodeVisibility,
              onToggleNodeLock: onToggleNodeLock,
            ),
          )
          .toList(growable: false),
    );
  }
}

class _SceneTreeNodeTile extends StatelessWidget {
  const _SceneTreeNodeTile({
    required this.node,
    required this.depth,
    required this.selectedNodeId,
    required this.enabled,
    required this.onSelectNode,
    required this.onToggleNodeVisibility,
    required this.onToggleNodeLock,
  });

  final AppSceneTreeNodeSnapshot node;
  final int depth;
  final BigInt? selectedNodeId;
  final bool enabled;
  final ValueChanged<BigInt> onSelectNode;
  final ValueChanged<BigInt> onToggleNodeVisibility;
  final ValueChanged<BigInt> onToggleNodeLock;

  @override
  Widget build(BuildContext context) {
    final isSelected = selectedNodeId == node.id;
    final theme = Theme.of(context);
    final labelColor = node.visible
        ? theme.colorScheme.onSurface
        : theme.colorScheme.onSurfaceVariant;

    return Padding(
      padding: EdgeInsets.only(
        left: depth * ShellTokens.sceneTreeIndent,
        bottom: ShellTokens.sceneTreeNodeGap,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          DecoratedBox(
            decoration: ShellSurfaceStyles.sceneTreeTile(
              context,
              selected: isSelected,
            ),
            child: Row(
              children: [
                Expanded(
                  child: InkWell(
                    key: ValueKey('scene-tree-node-${node.id}'),
                    borderRadius: BorderRadius.circular(
                      ShellTokens.surfaceRadius,
                    ),
                    onTap: enabled ? () => onSelectNode(node.id) : null,
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: ShellTokens.sceneTreeTileHorizontalPadding,
                        vertical: ShellTokens.sceneTreeTileVerticalPadding,
                      ),
                      child: Row(
                        children: [
                          Icon(
                            node.children.isEmpty
                                ? Icons.radio_button_unchecked
                                : Icons.account_tree_outlined,
                            size: 16,
                            color: labelColor,
                          ),
                          const SizedBox(width: ShellTokens.controlGap),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  node.name,
                                  style: theme.textTheme.bodyMedium?.copyWith(
                                    color: labelColor,
                                    fontWeight: isSelected
                                        ? FontWeight.w700
                                        : FontWeight.w500,
                                    decoration: node.visible
                                        ? TextDecoration.none
                                        : TextDecoration.lineThrough,
                                  ),
                                ),
                                Text(
                                  node.kindLabel,
                                  style: theme.textTheme.bodySmall?.copyWith(
                                    color: theme.colorScheme.onSurfaceVariant,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                Tooltip(
                  message: node.visible
                      ? 'Hide ${node.name}'
                      : 'Show ${node.name}',
                  child: IconButton(
                    key: ValueKey('scene-tree-visibility-${node.id}'),
                    onPressed: enabled
                        ? () => onToggleNodeVisibility(node.id)
                        : null,
                    icon: Icon(
                      node.visible ? Icons.visibility : Icons.visibility_off,
                    ),
                  ),
                ),
                Tooltip(
                  message: node.locked
                      ? 'Unlock ${node.name}'
                      : 'Lock ${node.name}',
                  child: IconButton(
                    key: ValueKey('scene-tree-lock-${node.id}'),
                    onPressed: enabled ? () => onToggleNodeLock(node.id) : null,
                    icon: Icon(node.locked ? Icons.lock : Icons.lock_open),
                  ),
                ),
              ],
              ),
            ),
          if (node.children.isNotEmpty) ...[
            const SizedBox(height: ShellTokens.sceneTreeNodeGap),
            ...node.children.map(
              (child) => _SceneTreeNodeTile(
                node: child,
                depth: depth + 1,
                selectedNodeId: selectedNodeId,
                enabled: enabled,
                onSelectNode: onSelectNode,
                onToggleNodeVisibility: onToggleNodeVisibility,
                onToggleNodeLock: onToggleNodeLock,
              ),
            ),
          ],
        ],
      ),
    );
  }
}
