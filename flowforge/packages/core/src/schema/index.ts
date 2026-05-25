/**
 * Schema Validation
 *
 * Zod schemas for validating FlowForge data structures.
 */

import { z } from 'zod';
import type { SemanticVersion, PrimitiveTypeName } from '../types/primitives.js';

// =============================================================================
// PRIMITIVE SCHEMAS
// =============================================================================

/**
 * Entity ID schema.
 */
export const EntityIdSchema = z.string().min(1);

/**
 * Semantic version schema.
 */
export const SemanticVersionSchema = z.string().regex(
  /^\d+\.\d+\.\d+(-[a-zA-Z0-9]+(\.[a-zA-Z0-9]+)*)?$/,
  'Invalid semantic version format'
) as z.ZodType<SemanticVersion>;

/**
 * Primitive type name schema.
 */
export const PrimitiveTypeNameSchema = z.enum([
  'ANY',
  'BOOLEAN',
  'INT',
  'FLOAT',
  'STRING',
  'ARRAY',
  'OBJECT',
  'BINARY',
  'NULL',
]) as z.ZodType<PrimitiveTypeName>;

/**
 * Position schema.
 */
export const PositionSchema = z.object({
  x: z.number(),
  y: z.number(),
});

/**
 * Size schema.
 */
export const SizeSchema = z.object({
  width: z.number().positive(),
  height: z.number().positive(),
});

/**
 * Rect schema.
 */
export const RectSchema = PositionSchema.merge(SizeSchema);

// =============================================================================
// TYPE DEFINITION SCHEMAS
// =============================================================================

/**
 * Type definition schema (recursive for nested types).
 */
interface TypeDefinitionShape {
  type: string;
  elementType?: TypeDefinitionShape;
  properties?: Record<string, TypeDefinitionShape>;
  nullable?: boolean;
  color?: string;
}

export const TypeDefinitionSchema: z.ZodType<TypeDefinitionShape> = z.lazy(() =>
  z.object({
    type: z.string(),
    elementType: TypeDefinitionSchema.optional(),
    properties: z.record(TypeDefinitionSchema).optional(),
    nullable: z.boolean().optional(),
    color: z.string().optional(),
  })
) as z.ZodType<TypeDefinitionShape>;

/**
 * Numeric constraint schema.
 */
export const NumericConstraintSchema = z.object({
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().positive().optional(),
  precision: z.number().int().nonnegative().optional(),
});

/**
 * String constraint schema.
 */
export const StringConstraintSchema = z.object({
  minLength: z.number().int().nonnegative().optional(),
  maxLength: z.number().int().positive().optional(),
  pattern: z.string().optional(),
  format: z.enum(['email', 'url', 'uuid', 'date', 'datetime', 'time']).optional(),
});

/**
 * Array constraint schema.
 */
export const ArrayConstraintSchema = z.object({
  minItems: z.number().int().nonnegative().optional(),
  maxItems: z.number().int().positive().optional(),
  uniqueItems: z.boolean().optional(),
});

// =============================================================================
// NODE DEFINITION SCHEMAS
// =============================================================================

/**
 * Input definition schema.
 */
export const InputDefinitionSchema = z.object({
  name: z.string(),
  type: z.union([TypeDefinitionSchema, z.string()]),
  required: z.boolean().optional(),
  defaultValue: z.unknown().optional(),
  description: z.string().optional(),
  multi: z.boolean().optional(),
  constraints: z.union([
    NumericConstraintSchema,
    StringConstraintSchema,
    ArrayConstraintSchema,
  ]).optional(),
  hidden: z.boolean().optional(),
});

/**
 * Output definition schema.
 */
export const OutputDefinitionSchema = z.object({
  name: z.string(),
  type: z.union([TypeDefinitionSchema, z.string()]),
  description: z.string().optional(),
  primary: z.boolean().optional(),
  hidden: z.boolean().optional(),
});

/**
 * Base widget schema.
 */
const BaseWidgetSchema = z.object({
  name: z.string(),
  label: z.string().optional(),
  description: z.string().optional(),
  disabled: z.boolean().optional(),
  hidden: z.boolean().optional(),
});

/**
 * Number widget schema.
 */
export const NumberWidgetSchema = BaseWidgetSchema.extend({
  type: z.literal('number'),
  min: z.number().optional(),
  max: z.number().optional(),
  step: z.number().optional(),
  precision: z.number().optional(),
  displayMode: z.enum(['slider', 'field', 'both']).optional(),
  defaultValue: z.number().optional(),
});

/**
 * Text widget schema.
 */
export const TextWidgetSchema = BaseWidgetSchema.extend({
  type: z.literal('text'),
  placeholder: z.string().optional(),
  multiline: z.boolean().optional(),
  rows: z.number().optional(),
  maxLength: z.number().optional(),
  defaultValue: z.string().optional(),
});

/**
 * Boolean widget schema.
 */
export const BooleanWidgetSchema = BaseWidgetSchema.extend({
  type: z.literal('boolean'),
  defaultValue: z.boolean().optional(),
});

/**
 * Select widget schema.
 */
export const SelectWidgetSchema = BaseWidgetSchema.extend({
  type: z.literal('select'),
  options: z.array(z.union([
    z.string(),
    z.object({ value: z.string(), label: z.string() }),
  ])),
  multiple: z.boolean().optional(),
  defaultValue: z.union([z.string(), z.array(z.string()).readonly()]).optional(),
});

/**
 * Widget definition schema.
 */
export const WidgetDefinitionSchema = z.discriminatedUnion('type', [
  NumberWidgetSchema,
  TextWidgetSchema,
  BooleanWidgetSchema,
  SelectWidgetSchema,
  BaseWidgetSchema.extend({
    type: z.literal('integer'),
    min: z.number().optional(),
    max: z.number().optional(),
    step: z.number().optional(),
    displayMode: z.enum(['slider', 'field', 'both']).optional(),
    defaultValue: z.number().optional(),
  }),
  BaseWidgetSchema.extend({
    type: z.literal('color'),
    format: z.enum(['hex', 'rgb', 'rgba', 'hsl']).optional(),
    defaultValue: z.string().optional(),
  }),
  BaseWidgetSchema.extend({
    type: z.literal('file'),
    accept: z.array(z.string()).readonly().optional(),
    directory: z.boolean().optional(),
    multiple: z.boolean().optional(),
    defaultValue: z.union([z.string(), z.array(z.string()).readonly()]).optional(),
  }),
  BaseWidgetSchema.extend({
    type: z.literal('custom'),
    component: z.string(),
    props: z.record(z.unknown()).optional(),
  }),
]);

/**
 * Node definition schema.
 */
export const NodeDefinitionSchema = z.object({
  type: z.string(),
  displayName: z.string().optional(),
  category: z.union([
    z.string(),
    z.object({
      path: z.string(),
      icon: z.string().optional(),
      description: z.string().optional(),
    }),
  ]),
  description: z.string().optional(),
  docsUrl: z.string().url().optional(),
  inputs: z.record(InputDefinitionSchema),
  outputs: z.record(OutputDefinitionSchema),
  widgets: z.record(WidgetDefinitionSchema).optional(),
  color: z.string().optional(),
  bgColor: z.string().optional(),
  size: SizeSchema.optional(),
  deprecated: z.boolean().optional(),
  replacedBy: z.string().optional(),
  executionMode: z.enum(['sync', 'async', 'stream']).optional(),
  tags: z.array(z.string()).readonly().optional(),
  providedBy: z.string().optional(),
  cacheable: z.boolean().optional(),
  executionCost: z.enum(['low', 'medium', 'high', 'very-high']).optional(),
});

// =============================================================================
// WORKFLOW SCHEMAS
// =============================================================================

/**
 * Node instance schema.
 */
export const NodeInstanceSchema = z.object({
  id: EntityIdSchema,
  type: z.string(),
  position: PositionSchema,
  size: SizeSchema.optional(),
  widgets: z.record(z.unknown()),
  title: z.string().optional(),
  order: z.number().optional(),
  disabled: z.boolean().optional(),
  collapsed: z.boolean().optional(),
  properties: z.record(z.unknown()).optional(),
  mode: z.enum(['default', 'bypass', 'mute']).optional(),
});

/**
 * Link endpoint schema.
 */
export const LinkEndpointSchema = z.object({
  nodeId: EntityIdSchema,
  slot: z.string(),
  slotIndex: z.number().int().nonnegative(),
});

/**
 * Link schema.
 */
export const LinkSchema = z.object({
  id: EntityIdSchema,
  source: LinkEndpointSchema,
  target: LinkEndpointSchema,
  type: z.string().optional(),
});

/**
 * Node group schema.
 */
export const NodeGroupSchema = z.object({
  id: EntityIdSchema,
  title: z.string(),
  bounds: RectSchema,
  color: z.string().optional(),
  fontSize: z.number().optional(),
  nodeIds: z.array(EntityIdSchema).readonly().optional(),
  locked: z.boolean().optional(),
  properties: z.record(z.unknown()).optional(),
});

/**
 * Workflow metadata schema.
 */
export const WorkflowMetadataSchema = z.object({
  title: z.string().optional(),
  description: z.string().optional(),
  author: z.union([
    z.object({
      name: z.string(),
      email: z.string().email().optional(),
      url: z.string().url().optional(),
    }),
    z.array(z.object({
      name: z.string(),
      email: z.string().email().optional(),
      url: z.string().url().optional(),
    })),
  ]).optional(),
  createdAt: z.string().datetime().optional(),
  updatedAt: z.string().datetime().optional(),
  version: SemanticVersionSchema.optional(),
  tags: z.array(z.string()).readonly().optional(),
  license: z.string().optional(),
  thumbnail: z.string().optional(),
  custom: z.record(z.unknown()).optional(),
});

/**
 * Workflow Zod validation schema.
 */
export const WorkflowZodSchema = z.object({
  version: SemanticVersionSchema,
  metadata: WorkflowMetadataSchema.optional(),
  config: z.object({
    view: z.object({
      offset: PositionSchema,
      scale: z.number().positive(),
    }).optional(),
    selectedNodes: z.array(EntityIdSchema).readonly().optional(),
    focusedNode: EntityIdSchema.optional(),
    showGrid: z.boolean().optional(),
    animateLinks: z.boolean().optional(),
    linkCurvature: z.number().min(0).max(1).optional(),
    snapToGrid: z.object({
      enabled: z.boolean(),
      size: z.number().positive(),
    }).optional(),
    custom: z.record(z.unknown()).optional(),
  }).optional(),
  nodes: z.record(NodeInstanceSchema),
  links: z.array(LinkSchema).readonly(),
  groups: z.array(NodeGroupSchema).readonly().optional(),
  reroutes: z.array(z.object({
    id: EntityIdSchema,
    position: PositionSchema,
    linkId: EntityIdSchema,
    type: z.string().optional(),
  })).readonly().optional(),
  requiredPlugins: z.array(z.string()).readonly().optional(),
  extra: z.record(z.unknown()).optional(),
});

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

/**
 * Validate a workflow against the schema.
 */
export function validateWorkflow(data: unknown): z.SafeParseReturnType<unknown, z.infer<typeof WorkflowZodSchema>> {
  return WorkflowZodSchema.safeParse(data);
}

/**
 * Validate a node definition against the schema.
 */
export function validateNodeDefinition(data: unknown): z.SafeParseReturnType<unknown, z.infer<typeof NodeDefinitionSchema>> {
  return NodeDefinitionSchema.safeParse(data);
}

/**
 * Parse and validate a workflow, throwing on error.
 */
export function parseWorkflow(data: unknown): z.infer<typeof WorkflowZodSchema> {
  return WorkflowZodSchema.parse(data);
}

/**
 * Parse and validate a node definition, throwing on error.
 */
export function parseNodeDefinition(data: unknown): z.infer<typeof NodeDefinitionSchema> {
  return NodeDefinitionSchema.parse(data);
}
