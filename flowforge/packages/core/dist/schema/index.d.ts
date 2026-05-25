/**
 * Schema Validation
 *
 * Zod schemas for validating FlowForge data structures.
 */
import { z } from 'zod';
import type { SemanticVersion, PrimitiveTypeName } from '../types/primitives.js';
/**
 * Entity ID schema.
 */
export declare const EntityIdSchema: z.ZodString;
/**
 * Semantic version schema.
 */
export declare const SemanticVersionSchema: z.ZodType<SemanticVersion>;
/**
 * Primitive type name schema.
 */
export declare const PrimitiveTypeNameSchema: z.ZodType<PrimitiveTypeName>;
/**
 * Position schema.
 */
export declare const PositionSchema: z.ZodObject<{
    x: z.ZodNumber;
    y: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    x: number;
    y: number;
}, {
    x: number;
    y: number;
}>;
/**
 * Size schema.
 */
export declare const SizeSchema: z.ZodObject<{
    width: z.ZodNumber;
    height: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    width: number;
    height: number;
}, {
    width: number;
    height: number;
}>;
/**
 * Rect schema.
 */
export declare const RectSchema: z.ZodObject<{
    x: z.ZodNumber;
    y: z.ZodNumber;
} & {
    width: z.ZodNumber;
    height: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    x: number;
    y: number;
    width: number;
    height: number;
}, {
    x: number;
    y: number;
    width: number;
    height: number;
}>;
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
export declare const TypeDefinitionSchema: z.ZodType<TypeDefinitionShape>;
/**
 * Numeric constraint schema.
 */
export declare const NumericConstraintSchema: z.ZodObject<{
    min: z.ZodOptional<z.ZodNumber>;
    max: z.ZodOptional<z.ZodNumber>;
    step: z.ZodOptional<z.ZodNumber>;
    precision: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
}, {
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
}>;
/**
 * String constraint schema.
 */
export declare const StringConstraintSchema: z.ZodObject<{
    minLength: z.ZodOptional<z.ZodNumber>;
    maxLength: z.ZodOptional<z.ZodNumber>;
    pattern: z.ZodOptional<z.ZodString>;
    format: z.ZodOptional<z.ZodEnum<["email", "url", "uuid", "date", "datetime", "time"]>>;
}, "strip", z.ZodTypeAny, {
    minLength?: number | undefined;
    maxLength?: number | undefined;
    pattern?: string | undefined;
    format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
}, {
    minLength?: number | undefined;
    maxLength?: number | undefined;
    pattern?: string | undefined;
    format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
}>;
/**
 * Array constraint schema.
 */
export declare const ArrayConstraintSchema: z.ZodObject<{
    minItems: z.ZodOptional<z.ZodNumber>;
    maxItems: z.ZodOptional<z.ZodNumber>;
    uniqueItems: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    minItems?: number | undefined;
    maxItems?: number | undefined;
    uniqueItems?: boolean | undefined;
}, {
    minItems?: number | undefined;
    maxItems?: number | undefined;
    uniqueItems?: boolean | undefined;
}>;
/**
 * Input definition schema.
 */
export declare const InputDefinitionSchema: z.ZodObject<{
    name: z.ZodString;
    type: z.ZodUnion<[z.ZodType<TypeDefinitionShape, z.ZodTypeDef, TypeDefinitionShape>, z.ZodString]>;
    required: z.ZodOptional<z.ZodBoolean>;
    defaultValue: z.ZodOptional<z.ZodUnknown>;
    description: z.ZodOptional<z.ZodString>;
    multi: z.ZodOptional<z.ZodBoolean>;
    constraints: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
        min: z.ZodOptional<z.ZodNumber>;
        max: z.ZodOptional<z.ZodNumber>;
        step: z.ZodOptional<z.ZodNumber>;
        precision: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
    }, {
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
    }>, z.ZodObject<{
        minLength: z.ZodOptional<z.ZodNumber>;
        maxLength: z.ZodOptional<z.ZodNumber>;
        pattern: z.ZodOptional<z.ZodString>;
        format: z.ZodOptional<z.ZodEnum<["email", "url", "uuid", "date", "datetime", "time"]>>;
    }, "strip", z.ZodTypeAny, {
        minLength?: number | undefined;
        maxLength?: number | undefined;
        pattern?: string | undefined;
        format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
    }, {
        minLength?: number | undefined;
        maxLength?: number | undefined;
        pattern?: string | undefined;
        format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
    }>, z.ZodObject<{
        minItems: z.ZodOptional<z.ZodNumber>;
        maxItems: z.ZodOptional<z.ZodNumber>;
        uniqueItems: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        minItems?: number | undefined;
        maxItems?: number | undefined;
        uniqueItems?: boolean | undefined;
    }, {
        minItems?: number | undefined;
        maxItems?: number | undefined;
        uniqueItems?: boolean | undefined;
    }>]>>;
    hidden: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    type: string | TypeDefinitionShape;
    name: string;
    required?: boolean | undefined;
    defaultValue?: unknown;
    description?: string | undefined;
    multi?: boolean | undefined;
    constraints?: {
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
    } | {
        minLength?: number | undefined;
        maxLength?: number | undefined;
        pattern?: string | undefined;
        format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
    } | {
        minItems?: number | undefined;
        maxItems?: number | undefined;
        uniqueItems?: boolean | undefined;
    } | undefined;
    hidden?: boolean | undefined;
}, {
    type: string | TypeDefinitionShape;
    name: string;
    required?: boolean | undefined;
    defaultValue?: unknown;
    description?: string | undefined;
    multi?: boolean | undefined;
    constraints?: {
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
    } | {
        minLength?: number | undefined;
        maxLength?: number | undefined;
        pattern?: string | undefined;
        format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
    } | {
        minItems?: number | undefined;
        maxItems?: number | undefined;
        uniqueItems?: boolean | undefined;
    } | undefined;
    hidden?: boolean | undefined;
}>;
/**
 * Output definition schema.
 */
export declare const OutputDefinitionSchema: z.ZodObject<{
    name: z.ZodString;
    type: z.ZodUnion<[z.ZodType<TypeDefinitionShape, z.ZodTypeDef, TypeDefinitionShape>, z.ZodString]>;
    description: z.ZodOptional<z.ZodString>;
    primary: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    type: string | TypeDefinitionShape;
    name: string;
    description?: string | undefined;
    hidden?: boolean | undefined;
    primary?: boolean | undefined;
}, {
    type: string | TypeDefinitionShape;
    name: string;
    description?: string | undefined;
    hidden?: boolean | undefined;
    primary?: boolean | undefined;
}>;
/**
 * Number widget schema.
 */
export declare const NumberWidgetSchema: z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"number">;
    min: z.ZodOptional<z.ZodNumber>;
    max: z.ZodOptional<z.ZodNumber>;
    step: z.ZodOptional<z.ZodNumber>;
    precision: z.ZodOptional<z.ZodNumber>;
    displayMode: z.ZodOptional<z.ZodEnum<["slider", "field", "both"]>>;
    defaultValue: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    type: "number";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}, {
    type: "number";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}>;
/**
 * Text widget schema.
 */
export declare const TextWidgetSchema: z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"text">;
    placeholder: z.ZodOptional<z.ZodString>;
    multiline: z.ZodOptional<z.ZodBoolean>;
    rows: z.ZodOptional<z.ZodNumber>;
    maxLength: z.ZodOptional<z.ZodNumber>;
    defaultValue: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "text";
    name: string;
    disabled?: boolean | undefined;
    maxLength?: number | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    placeholder?: string | undefined;
    multiline?: boolean | undefined;
    rows?: number | undefined;
}, {
    type: "text";
    name: string;
    disabled?: boolean | undefined;
    maxLength?: number | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    placeholder?: string | undefined;
    multiline?: boolean | undefined;
    rows?: number | undefined;
}>;
/**
 * Boolean widget schema.
 */
export declare const BooleanWidgetSchema: z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"boolean">;
    defaultValue: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    type: "boolean";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}, {
    type: "boolean";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}>;
/**
 * Select widget schema.
 */
export declare const SelectWidgetSchema: z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"select">;
    options: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodObject<{
        value: z.ZodString;
        label: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        value: string;
        label: string;
    }, {
        value: string;
        label: string;
    }>]>, "many">;
    multiple: z.ZodOptional<z.ZodBoolean>;
    defaultValue: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodReadonly<z.ZodArray<z.ZodString, "many">>]>>;
}, "strip", z.ZodTypeAny, {
    options: (string | {
        value: string;
        label: string;
    })[];
    type: "select";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
}, {
    options: (string | {
        value: string;
        label: string;
    })[];
    type: "select";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
}>;
/**
 * Widget definition schema.
 */
export declare const WidgetDefinitionSchema: z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"number">;
    min: z.ZodOptional<z.ZodNumber>;
    max: z.ZodOptional<z.ZodNumber>;
    step: z.ZodOptional<z.ZodNumber>;
    precision: z.ZodOptional<z.ZodNumber>;
    displayMode: z.ZodOptional<z.ZodEnum<["slider", "field", "both"]>>;
    defaultValue: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    type: "number";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}, {
    type: "number";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    precision?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"text">;
    placeholder: z.ZodOptional<z.ZodString>;
    multiline: z.ZodOptional<z.ZodBoolean>;
    rows: z.ZodOptional<z.ZodNumber>;
    maxLength: z.ZodOptional<z.ZodNumber>;
    defaultValue: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "text";
    name: string;
    disabled?: boolean | undefined;
    maxLength?: number | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    placeholder?: string | undefined;
    multiline?: boolean | undefined;
    rows?: number | undefined;
}, {
    type: "text";
    name: string;
    disabled?: boolean | undefined;
    maxLength?: number | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    placeholder?: string | undefined;
    multiline?: boolean | undefined;
    rows?: number | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"boolean">;
    defaultValue: z.ZodOptional<z.ZodBoolean>;
}, "strip", z.ZodTypeAny, {
    type: "boolean";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}, {
    type: "boolean";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"select">;
    options: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodObject<{
        value: z.ZodString;
        label: z.ZodString;
    }, "strip", z.ZodTypeAny, {
        value: string;
        label: string;
    }, {
        value: string;
        label: string;
    }>]>, "many">;
    multiple: z.ZodOptional<z.ZodBoolean>;
    defaultValue: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodReadonly<z.ZodArray<z.ZodString, "many">>]>>;
}, "strip", z.ZodTypeAny, {
    options: (string | {
        value: string;
        label: string;
    })[];
    type: "select";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
}, {
    options: (string | {
        value: string;
        label: string;
    })[];
    type: "select";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"integer">;
    min: z.ZodOptional<z.ZodNumber>;
    max: z.ZodOptional<z.ZodNumber>;
    step: z.ZodOptional<z.ZodNumber>;
    displayMode: z.ZodOptional<z.ZodEnum<["slider", "field", "both"]>>;
    defaultValue: z.ZodOptional<z.ZodNumber>;
}, "strip", z.ZodTypeAny, {
    type: "integer";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}, {
    type: "integer";
    name: string;
    disabled?: boolean | undefined;
    min?: number | undefined;
    max?: number | undefined;
    step?: number | undefined;
    defaultValue?: number | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    displayMode?: "slider" | "field" | "both" | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"color">;
    format: z.ZodOptional<z.ZodEnum<["hex", "rgb", "rgba", "hsl"]>>;
    defaultValue: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    type: "color";
    name: string;
    disabled?: boolean | undefined;
    format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}, {
    type: "color";
    name: string;
    disabled?: boolean | undefined;
    format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
    defaultValue?: string | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"file">;
    accept: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
    directory: z.ZodOptional<z.ZodBoolean>;
    multiple: z.ZodOptional<z.ZodBoolean>;
    defaultValue: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodReadonly<z.ZodArray<z.ZodString, "many">>]>>;
}, "strip", z.ZodTypeAny, {
    type: "file";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
    accept?: readonly string[] | undefined;
    directory?: boolean | undefined;
}, {
    type: "file";
    name: string;
    disabled?: boolean | undefined;
    defaultValue?: string | readonly string[] | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    multiple?: boolean | undefined;
    accept?: readonly string[] | undefined;
    directory?: boolean | undefined;
}>, z.ZodObject<{
    name: z.ZodString;
    label: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    hidden: z.ZodOptional<z.ZodBoolean>;
} & {
    type: z.ZodLiteral<"custom">;
    component: z.ZodString;
    props: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    type: "custom";
    name: string;
    component: string;
    disabled?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    props?: Record<string, unknown> | undefined;
}, {
    type: "custom";
    name: string;
    component: string;
    disabled?: boolean | undefined;
    description?: string | undefined;
    hidden?: boolean | undefined;
    label?: string | undefined;
    props?: Record<string, unknown> | undefined;
}>]>;
/**
 * Node definition schema.
 */
export declare const NodeDefinitionSchema: z.ZodObject<{
    type: z.ZodString;
    displayName: z.ZodOptional<z.ZodString>;
    category: z.ZodUnion<[z.ZodString, z.ZodObject<{
        path: z.ZodString;
        icon: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        path: string;
        description?: string | undefined;
        icon?: string | undefined;
    }, {
        path: string;
        description?: string | undefined;
        icon?: string | undefined;
    }>]>;
    description: z.ZodOptional<z.ZodString>;
    docsUrl: z.ZodOptional<z.ZodString>;
    inputs: z.ZodRecord<z.ZodString, z.ZodObject<{
        name: z.ZodString;
        type: z.ZodUnion<[z.ZodType<TypeDefinitionShape, z.ZodTypeDef, TypeDefinitionShape>, z.ZodString]>;
        required: z.ZodOptional<z.ZodBoolean>;
        defaultValue: z.ZodOptional<z.ZodUnknown>;
        description: z.ZodOptional<z.ZodString>;
        multi: z.ZodOptional<z.ZodBoolean>;
        constraints: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
            min: z.ZodOptional<z.ZodNumber>;
            max: z.ZodOptional<z.ZodNumber>;
            step: z.ZodOptional<z.ZodNumber>;
            precision: z.ZodOptional<z.ZodNumber>;
        }, "strip", z.ZodTypeAny, {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        }, {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        }>, z.ZodObject<{
            minLength: z.ZodOptional<z.ZodNumber>;
            maxLength: z.ZodOptional<z.ZodNumber>;
            pattern: z.ZodOptional<z.ZodString>;
            format: z.ZodOptional<z.ZodEnum<["email", "url", "uuid", "date", "datetime", "time"]>>;
        }, "strip", z.ZodTypeAny, {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        }, {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        }>, z.ZodObject<{
            minItems: z.ZodOptional<z.ZodNumber>;
            maxItems: z.ZodOptional<z.ZodNumber>;
            uniqueItems: z.ZodOptional<z.ZodBoolean>;
        }, "strip", z.ZodTypeAny, {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        }, {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        }>]>>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        type: string | TypeDefinitionShape;
        name: string;
        required?: boolean | undefined;
        defaultValue?: unknown;
        description?: string | undefined;
        multi?: boolean | undefined;
        constraints?: {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        } | {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        } | {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        } | undefined;
        hidden?: boolean | undefined;
    }, {
        type: string | TypeDefinitionShape;
        name: string;
        required?: boolean | undefined;
        defaultValue?: unknown;
        description?: string | undefined;
        multi?: boolean | undefined;
        constraints?: {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        } | {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        } | {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        } | undefined;
        hidden?: boolean | undefined;
    }>>;
    outputs: z.ZodRecord<z.ZodString, z.ZodObject<{
        name: z.ZodString;
        type: z.ZodUnion<[z.ZodType<TypeDefinitionShape, z.ZodTypeDef, TypeDefinitionShape>, z.ZodString]>;
        description: z.ZodOptional<z.ZodString>;
        primary: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        type: string | TypeDefinitionShape;
        name: string;
        description?: string | undefined;
        hidden?: boolean | undefined;
        primary?: boolean | undefined;
    }, {
        type: string | TypeDefinitionShape;
        name: string;
        description?: string | undefined;
        hidden?: boolean | undefined;
        primary?: boolean | undefined;
    }>>;
    widgets: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodDiscriminatedUnion<"type", [z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"number">;
        min: z.ZodOptional<z.ZodNumber>;
        max: z.ZodOptional<z.ZodNumber>;
        step: z.ZodOptional<z.ZodNumber>;
        precision: z.ZodOptional<z.ZodNumber>;
        displayMode: z.ZodOptional<z.ZodEnum<["slider", "field", "both"]>>;
        defaultValue: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        type: "number";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    }, {
        type: "number";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"text">;
        placeholder: z.ZodOptional<z.ZodString>;
        multiline: z.ZodOptional<z.ZodBoolean>;
        rows: z.ZodOptional<z.ZodNumber>;
        maxLength: z.ZodOptional<z.ZodNumber>;
        defaultValue: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "text";
        name: string;
        disabled?: boolean | undefined;
        maxLength?: number | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        placeholder?: string | undefined;
        multiline?: boolean | undefined;
        rows?: number | undefined;
    }, {
        type: "text";
        name: string;
        disabled?: boolean | undefined;
        maxLength?: number | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        placeholder?: string | undefined;
        multiline?: boolean | undefined;
        rows?: number | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"boolean">;
        defaultValue: z.ZodOptional<z.ZodBoolean>;
    }, "strip", z.ZodTypeAny, {
        type: "boolean";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    }, {
        type: "boolean";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"select">;
        options: z.ZodArray<z.ZodUnion<[z.ZodString, z.ZodObject<{
            value: z.ZodString;
            label: z.ZodString;
        }, "strip", z.ZodTypeAny, {
            value: string;
            label: string;
        }, {
            value: string;
            label: string;
        }>]>, "many">;
        multiple: z.ZodOptional<z.ZodBoolean>;
        defaultValue: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodReadonly<z.ZodArray<z.ZodString, "many">>]>>;
    }, "strip", z.ZodTypeAny, {
        options: (string | {
            value: string;
            label: string;
        })[];
        type: "select";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
    }, {
        options: (string | {
            value: string;
            label: string;
        })[];
        type: "select";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"integer">;
        min: z.ZodOptional<z.ZodNumber>;
        max: z.ZodOptional<z.ZodNumber>;
        step: z.ZodOptional<z.ZodNumber>;
        displayMode: z.ZodOptional<z.ZodEnum<["slider", "field", "both"]>>;
        defaultValue: z.ZodOptional<z.ZodNumber>;
    }, "strip", z.ZodTypeAny, {
        type: "integer";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    }, {
        type: "integer";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"color">;
        format: z.ZodOptional<z.ZodEnum<["hex", "rgb", "rgba", "hsl"]>>;
        defaultValue: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        type: "color";
        name: string;
        disabled?: boolean | undefined;
        format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    }, {
        type: "color";
        name: string;
        disabled?: boolean | undefined;
        format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"file">;
        accept: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
        directory: z.ZodOptional<z.ZodBoolean>;
        multiple: z.ZodOptional<z.ZodBoolean>;
        defaultValue: z.ZodOptional<z.ZodUnion<[z.ZodString, z.ZodReadonly<z.ZodArray<z.ZodString, "many">>]>>;
    }, "strip", z.ZodTypeAny, {
        type: "file";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
        accept?: readonly string[] | undefined;
        directory?: boolean | undefined;
    }, {
        type: "file";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
        accept?: readonly string[] | undefined;
        directory?: boolean | undefined;
    }>, z.ZodObject<{
        name: z.ZodString;
        label: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        hidden: z.ZodOptional<z.ZodBoolean>;
    } & {
        type: z.ZodLiteral<"custom">;
        component: z.ZodString;
        props: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        type: "custom";
        name: string;
        component: string;
        disabled?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        props?: Record<string, unknown> | undefined;
    }, {
        type: "custom";
        name: string;
        component: string;
        disabled?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        props?: Record<string, unknown> | undefined;
    }>]>>>;
    color: z.ZodOptional<z.ZodString>;
    bgColor: z.ZodOptional<z.ZodString>;
    size: z.ZodOptional<z.ZodObject<{
        width: z.ZodNumber;
        height: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        width: number;
        height: number;
    }, {
        width: number;
        height: number;
    }>>;
    deprecated: z.ZodOptional<z.ZodBoolean>;
    replacedBy: z.ZodOptional<z.ZodString>;
    executionMode: z.ZodOptional<z.ZodEnum<["sync", "async", "stream"]>>;
    tags: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
    providedBy: z.ZodOptional<z.ZodString>;
    cacheable: z.ZodOptional<z.ZodBoolean>;
    executionCost: z.ZodOptional<z.ZodEnum<["low", "medium", "high", "very-high"]>>;
}, "strip", z.ZodTypeAny, {
    type: string;
    category: string | {
        path: string;
        description?: string | undefined;
        icon?: string | undefined;
    };
    inputs: Record<string, {
        type: string | TypeDefinitionShape;
        name: string;
        required?: boolean | undefined;
        defaultValue?: unknown;
        description?: string | undefined;
        multi?: boolean | undefined;
        constraints?: {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        } | {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        } | {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        } | undefined;
        hidden?: boolean | undefined;
    }>;
    outputs: Record<string, {
        type: string | TypeDefinitionShape;
        name: string;
        description?: string | undefined;
        hidden?: boolean | undefined;
        primary?: boolean | undefined;
    }>;
    color?: string | undefined;
    description?: string | undefined;
    displayName?: string | undefined;
    docsUrl?: string | undefined;
    widgets?: Record<string, {
        type: "number";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    } | {
        type: "text";
        name: string;
        disabled?: boolean | undefined;
        maxLength?: number | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        placeholder?: string | undefined;
        multiline?: boolean | undefined;
        rows?: number | undefined;
    } | {
        type: "boolean";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    } | {
        options: (string | {
            value: string;
            label: string;
        })[];
        type: "select";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
    } | {
        type: "integer";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    } | {
        type: "color";
        name: string;
        disabled?: boolean | undefined;
        format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    } | {
        type: "file";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
        accept?: readonly string[] | undefined;
        directory?: boolean | undefined;
    } | {
        type: "custom";
        name: string;
        component: string;
        disabled?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        props?: Record<string, unknown> | undefined;
    }> | undefined;
    bgColor?: string | undefined;
    size?: {
        width: number;
        height: number;
    } | undefined;
    deprecated?: boolean | undefined;
    replacedBy?: string | undefined;
    executionMode?: "sync" | "async" | "stream" | undefined;
    tags?: readonly string[] | undefined;
    providedBy?: string | undefined;
    cacheable?: boolean | undefined;
    executionCost?: "low" | "medium" | "high" | "very-high" | undefined;
}, {
    type: string;
    category: string | {
        path: string;
        description?: string | undefined;
        icon?: string | undefined;
    };
    inputs: Record<string, {
        type: string | TypeDefinitionShape;
        name: string;
        required?: boolean | undefined;
        defaultValue?: unknown;
        description?: string | undefined;
        multi?: boolean | undefined;
        constraints?: {
            min?: number | undefined;
            max?: number | undefined;
            step?: number | undefined;
            precision?: number | undefined;
        } | {
            minLength?: number | undefined;
            maxLength?: number | undefined;
            pattern?: string | undefined;
            format?: "email" | "url" | "uuid" | "date" | "datetime" | "time" | undefined;
        } | {
            minItems?: number | undefined;
            maxItems?: number | undefined;
            uniqueItems?: boolean | undefined;
        } | undefined;
        hidden?: boolean | undefined;
    }>;
    outputs: Record<string, {
        type: string | TypeDefinitionShape;
        name: string;
        description?: string | undefined;
        hidden?: boolean | undefined;
        primary?: boolean | undefined;
    }>;
    color?: string | undefined;
    description?: string | undefined;
    displayName?: string | undefined;
    docsUrl?: string | undefined;
    widgets?: Record<string, {
        type: "number";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        precision?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    } | {
        type: "text";
        name: string;
        disabled?: boolean | undefined;
        maxLength?: number | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        placeholder?: string | undefined;
        multiline?: boolean | undefined;
        rows?: number | undefined;
    } | {
        type: "boolean";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    } | {
        options: (string | {
            value: string;
            label: string;
        })[];
        type: "select";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
    } | {
        type: "integer";
        name: string;
        disabled?: boolean | undefined;
        min?: number | undefined;
        max?: number | undefined;
        step?: number | undefined;
        defaultValue?: number | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        displayMode?: "slider" | "field" | "both" | undefined;
    } | {
        type: "color";
        name: string;
        disabled?: boolean | undefined;
        format?: "hex" | "rgb" | "rgba" | "hsl" | undefined;
        defaultValue?: string | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
    } | {
        type: "file";
        name: string;
        disabled?: boolean | undefined;
        defaultValue?: string | readonly string[] | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        multiple?: boolean | undefined;
        accept?: readonly string[] | undefined;
        directory?: boolean | undefined;
    } | {
        type: "custom";
        name: string;
        component: string;
        disabled?: boolean | undefined;
        description?: string | undefined;
        hidden?: boolean | undefined;
        label?: string | undefined;
        props?: Record<string, unknown> | undefined;
    }> | undefined;
    bgColor?: string | undefined;
    size?: {
        width: number;
        height: number;
    } | undefined;
    deprecated?: boolean | undefined;
    replacedBy?: string | undefined;
    executionMode?: "sync" | "async" | "stream" | undefined;
    tags?: readonly string[] | undefined;
    providedBy?: string | undefined;
    cacheable?: boolean | undefined;
    executionCost?: "low" | "medium" | "high" | "very-high" | undefined;
}>;
/**
 * Node instance schema.
 */
export declare const NodeInstanceSchema: z.ZodObject<{
    id: z.ZodString;
    type: z.ZodString;
    position: z.ZodObject<{
        x: z.ZodNumber;
        y: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        x: number;
        y: number;
    }, {
        x: number;
        y: number;
    }>;
    size: z.ZodOptional<z.ZodObject<{
        width: z.ZodNumber;
        height: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        width: number;
        height: number;
    }, {
        width: number;
        height: number;
    }>>;
    widgets: z.ZodRecord<z.ZodString, z.ZodUnknown>;
    title: z.ZodOptional<z.ZodString>;
    order: z.ZodOptional<z.ZodNumber>;
    disabled: z.ZodOptional<z.ZodBoolean>;
    collapsed: z.ZodOptional<z.ZodBoolean>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    mode: z.ZodOptional<z.ZodEnum<["default", "bypass", "mute"]>>;
}, "strip", z.ZodTypeAny, {
    type: string;
    widgets: Record<string, unknown>;
    id: string;
    position: {
        x: number;
        y: number;
    };
    disabled?: boolean | undefined;
    properties?: Record<string, unknown> | undefined;
    size?: {
        width: number;
        height: number;
    } | undefined;
    title?: string | undefined;
    order?: number | undefined;
    collapsed?: boolean | undefined;
    mode?: "default" | "bypass" | "mute" | undefined;
}, {
    type: string;
    widgets: Record<string, unknown>;
    id: string;
    position: {
        x: number;
        y: number;
    };
    disabled?: boolean | undefined;
    properties?: Record<string, unknown> | undefined;
    size?: {
        width: number;
        height: number;
    } | undefined;
    title?: string | undefined;
    order?: number | undefined;
    collapsed?: boolean | undefined;
    mode?: "default" | "bypass" | "mute" | undefined;
}>;
/**
 * Link endpoint schema.
 */
export declare const LinkEndpointSchema: z.ZodObject<{
    nodeId: z.ZodString;
    slot: z.ZodString;
    slotIndex: z.ZodNumber;
}, "strip", z.ZodTypeAny, {
    nodeId: string;
    slot: string;
    slotIndex: number;
}, {
    nodeId: string;
    slot: string;
    slotIndex: number;
}>;
/**
 * Link schema.
 */
export declare const LinkSchema: z.ZodObject<{
    id: z.ZodString;
    source: z.ZodObject<{
        nodeId: z.ZodString;
        slot: z.ZodString;
        slotIndex: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        nodeId: string;
        slot: string;
        slotIndex: number;
    }, {
        nodeId: string;
        slot: string;
        slotIndex: number;
    }>;
    target: z.ZodObject<{
        nodeId: z.ZodString;
        slot: z.ZodString;
        slotIndex: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        nodeId: string;
        slot: string;
        slotIndex: number;
    }, {
        nodeId: string;
        slot: string;
        slotIndex: number;
    }>;
    type: z.ZodOptional<z.ZodString>;
}, "strip", z.ZodTypeAny, {
    id: string;
    source: {
        nodeId: string;
        slot: string;
        slotIndex: number;
    };
    target: {
        nodeId: string;
        slot: string;
        slotIndex: number;
    };
    type?: string | undefined;
}, {
    id: string;
    source: {
        nodeId: string;
        slot: string;
        slotIndex: number;
    };
    target: {
        nodeId: string;
        slot: string;
        slotIndex: number;
    };
    type?: string | undefined;
}>;
/**
 * Node group schema.
 */
export declare const NodeGroupSchema: z.ZodObject<{
    id: z.ZodString;
    title: z.ZodString;
    bounds: z.ZodObject<{
        x: z.ZodNumber;
        y: z.ZodNumber;
    } & {
        width: z.ZodNumber;
        height: z.ZodNumber;
    }, "strip", z.ZodTypeAny, {
        x: number;
        y: number;
        width: number;
        height: number;
    }, {
        x: number;
        y: number;
        width: number;
        height: number;
    }>;
    color: z.ZodOptional<z.ZodString>;
    fontSize: z.ZodOptional<z.ZodNumber>;
    nodeIds: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
    locked: z.ZodOptional<z.ZodBoolean>;
    properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    id: string;
    title: string;
    bounds: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    color?: string | undefined;
    properties?: Record<string, unknown> | undefined;
    fontSize?: number | undefined;
    nodeIds?: readonly string[] | undefined;
    locked?: boolean | undefined;
}, {
    id: string;
    title: string;
    bounds: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    color?: string | undefined;
    properties?: Record<string, unknown> | undefined;
    fontSize?: number | undefined;
    nodeIds?: readonly string[] | undefined;
    locked?: boolean | undefined;
}>;
/**
 * Workflow metadata schema.
 */
export declare const WorkflowMetadataSchema: z.ZodObject<{
    title: z.ZodOptional<z.ZodString>;
    description: z.ZodOptional<z.ZodString>;
    author: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
        name: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }, {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }>, z.ZodArray<z.ZodObject<{
        name: z.ZodString;
        email: z.ZodOptional<z.ZodString>;
        url: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }, {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }>, "many">]>>;
    createdAt: z.ZodOptional<z.ZodString>;
    updatedAt: z.ZodOptional<z.ZodString>;
    version: z.ZodOptional<z.ZodType<`${number}.${number}.${number}${string}`, z.ZodTypeDef, `${number}.${number}.${number}${string}`>>;
    tags: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
    license: z.ZodOptional<z.ZodString>;
    thumbnail: z.ZodOptional<z.ZodString>;
    custom: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    custom?: Record<string, unknown> | undefined;
    description?: string | undefined;
    tags?: readonly string[] | undefined;
    title?: string | undefined;
    author?: {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    } | {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }[] | undefined;
    createdAt?: string | undefined;
    updatedAt?: string | undefined;
    version?: `${number}.${number}.${number}${string}` | undefined;
    license?: string | undefined;
    thumbnail?: string | undefined;
}, {
    custom?: Record<string, unknown> | undefined;
    description?: string | undefined;
    tags?: readonly string[] | undefined;
    title?: string | undefined;
    author?: {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    } | {
        name: string;
        email?: string | undefined;
        url?: string | undefined;
    }[] | undefined;
    createdAt?: string | undefined;
    updatedAt?: string | undefined;
    version?: `${number}.${number}.${number}${string}` | undefined;
    license?: string | undefined;
    thumbnail?: string | undefined;
}>;
/**
 * Workflow Zod validation schema.
 */
export declare const WorkflowZodSchema: z.ZodObject<{
    version: z.ZodType<`${number}.${number}.${number}${string}`, z.ZodTypeDef, `${number}.${number}.${number}${string}`>;
    metadata: z.ZodOptional<z.ZodObject<{
        title: z.ZodOptional<z.ZodString>;
        description: z.ZodOptional<z.ZodString>;
        author: z.ZodOptional<z.ZodUnion<[z.ZodObject<{
            name: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
            url: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }, {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }>, z.ZodArray<z.ZodObject<{
            name: z.ZodString;
            email: z.ZodOptional<z.ZodString>;
            url: z.ZodOptional<z.ZodString>;
        }, "strip", z.ZodTypeAny, {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }, {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }>, "many">]>>;
        createdAt: z.ZodOptional<z.ZodString>;
        updatedAt: z.ZodOptional<z.ZodString>;
        version: z.ZodOptional<z.ZodType<`${number}.${number}.${number}${string}`, z.ZodTypeDef, `${number}.${number}.${number}${string}`>>;
        tags: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
        license: z.ZodOptional<z.ZodString>;
        thumbnail: z.ZodOptional<z.ZodString>;
        custom: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        custom?: Record<string, unknown> | undefined;
        description?: string | undefined;
        tags?: readonly string[] | undefined;
        title?: string | undefined;
        author?: {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        } | {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }[] | undefined;
        createdAt?: string | undefined;
        updatedAt?: string | undefined;
        version?: `${number}.${number}.${number}${string}` | undefined;
        license?: string | undefined;
        thumbnail?: string | undefined;
    }, {
        custom?: Record<string, unknown> | undefined;
        description?: string | undefined;
        tags?: readonly string[] | undefined;
        title?: string | undefined;
        author?: {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        } | {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }[] | undefined;
        createdAt?: string | undefined;
        updatedAt?: string | undefined;
        version?: `${number}.${number}.${number}${string}` | undefined;
        license?: string | undefined;
        thumbnail?: string | undefined;
    }>>;
    config: z.ZodOptional<z.ZodObject<{
        view: z.ZodOptional<z.ZodObject<{
            offset: z.ZodObject<{
                x: z.ZodNumber;
                y: z.ZodNumber;
            }, "strip", z.ZodTypeAny, {
                x: number;
                y: number;
            }, {
                x: number;
                y: number;
            }>;
            scale: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        }, {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        }>>;
        selectedNodes: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
        focusedNode: z.ZodOptional<z.ZodString>;
        showGrid: z.ZodOptional<z.ZodBoolean>;
        animateLinks: z.ZodOptional<z.ZodBoolean>;
        linkCurvature: z.ZodOptional<z.ZodNumber>;
        snapToGrid: z.ZodOptional<z.ZodObject<{
            enabled: z.ZodBoolean;
            size: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            size: number;
            enabled: boolean;
        }, {
            size: number;
            enabled: boolean;
        }>>;
        custom: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        custom?: Record<string, unknown> | undefined;
        view?: {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        } | undefined;
        selectedNodes?: readonly string[] | undefined;
        focusedNode?: string | undefined;
        showGrid?: boolean | undefined;
        animateLinks?: boolean | undefined;
        linkCurvature?: number | undefined;
        snapToGrid?: {
            size: number;
            enabled: boolean;
        } | undefined;
    }, {
        custom?: Record<string, unknown> | undefined;
        view?: {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        } | undefined;
        selectedNodes?: readonly string[] | undefined;
        focusedNode?: string | undefined;
        showGrid?: boolean | undefined;
        animateLinks?: boolean | undefined;
        linkCurvature?: number | undefined;
        snapToGrid?: {
            size: number;
            enabled: boolean;
        } | undefined;
    }>>;
    nodes: z.ZodRecord<z.ZodString, z.ZodObject<{
        id: z.ZodString;
        type: z.ZodString;
        position: z.ZodObject<{
            x: z.ZodNumber;
            y: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            x: number;
            y: number;
        }, {
            x: number;
            y: number;
        }>;
        size: z.ZodOptional<z.ZodObject<{
            width: z.ZodNumber;
            height: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            width: number;
            height: number;
        }, {
            width: number;
            height: number;
        }>>;
        widgets: z.ZodRecord<z.ZodString, z.ZodUnknown>;
        title: z.ZodOptional<z.ZodString>;
        order: z.ZodOptional<z.ZodNumber>;
        disabled: z.ZodOptional<z.ZodBoolean>;
        collapsed: z.ZodOptional<z.ZodBoolean>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
        mode: z.ZodOptional<z.ZodEnum<["default", "bypass", "mute"]>>;
    }, "strip", z.ZodTypeAny, {
        type: string;
        widgets: Record<string, unknown>;
        id: string;
        position: {
            x: number;
            y: number;
        };
        disabled?: boolean | undefined;
        properties?: Record<string, unknown> | undefined;
        size?: {
            width: number;
            height: number;
        } | undefined;
        title?: string | undefined;
        order?: number | undefined;
        collapsed?: boolean | undefined;
        mode?: "default" | "bypass" | "mute" | undefined;
    }, {
        type: string;
        widgets: Record<string, unknown>;
        id: string;
        position: {
            x: number;
            y: number;
        };
        disabled?: boolean | undefined;
        properties?: Record<string, unknown> | undefined;
        size?: {
            width: number;
            height: number;
        } | undefined;
        title?: string | undefined;
        order?: number | undefined;
        collapsed?: boolean | undefined;
        mode?: "default" | "bypass" | "mute" | undefined;
    }>>;
    links: z.ZodReadonly<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        source: z.ZodObject<{
            nodeId: z.ZodString;
            slot: z.ZodString;
            slotIndex: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            nodeId: string;
            slot: string;
            slotIndex: number;
        }, {
            nodeId: string;
            slot: string;
            slotIndex: number;
        }>;
        target: z.ZodObject<{
            nodeId: z.ZodString;
            slot: z.ZodString;
            slotIndex: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            nodeId: string;
            slot: string;
            slotIndex: number;
        }, {
            nodeId: string;
            slot: string;
            slotIndex: number;
        }>;
        type: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        source: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        target: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        type?: string | undefined;
    }, {
        id: string;
        source: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        target: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        type?: string | undefined;
    }>, "many">>;
    groups: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        title: z.ZodString;
        bounds: z.ZodObject<{
            x: z.ZodNumber;
            y: z.ZodNumber;
        } & {
            width: z.ZodNumber;
            height: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            x: number;
            y: number;
            width: number;
            height: number;
        }, {
            x: number;
            y: number;
            width: number;
            height: number;
        }>;
        color: z.ZodOptional<z.ZodString>;
        fontSize: z.ZodOptional<z.ZodNumber>;
        nodeIds: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
        locked: z.ZodOptional<z.ZodBoolean>;
        properties: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        title: string;
        bounds: {
            x: number;
            y: number;
            width: number;
            height: number;
        };
        color?: string | undefined;
        properties?: Record<string, unknown> | undefined;
        fontSize?: number | undefined;
        nodeIds?: readonly string[] | undefined;
        locked?: boolean | undefined;
    }, {
        id: string;
        title: string;
        bounds: {
            x: number;
            y: number;
            width: number;
            height: number;
        };
        color?: string | undefined;
        properties?: Record<string, unknown> | undefined;
        fontSize?: number | undefined;
        nodeIds?: readonly string[] | undefined;
        locked?: boolean | undefined;
    }>, "many">>>;
    reroutes: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodObject<{
        id: z.ZodString;
        position: z.ZodObject<{
            x: z.ZodNumber;
            y: z.ZodNumber;
        }, "strip", z.ZodTypeAny, {
            x: number;
            y: number;
        }, {
            x: number;
            y: number;
        }>;
        linkId: z.ZodString;
        type: z.ZodOptional<z.ZodString>;
    }, "strip", z.ZodTypeAny, {
        id: string;
        position: {
            x: number;
            y: number;
        };
        linkId: string;
        type?: string | undefined;
    }, {
        id: string;
        position: {
            x: number;
            y: number;
        };
        linkId: string;
        type?: string | undefined;
    }>, "many">>>;
    requiredPlugins: z.ZodOptional<z.ZodReadonly<z.ZodArray<z.ZodString, "many">>>;
    extra: z.ZodOptional<z.ZodRecord<z.ZodString, z.ZodUnknown>>;
}, "strip", z.ZodTypeAny, {
    version: `${number}.${number}.${number}${string}`;
    nodes: Record<string, {
        type: string;
        widgets: Record<string, unknown>;
        id: string;
        position: {
            x: number;
            y: number;
        };
        disabled?: boolean | undefined;
        properties?: Record<string, unknown> | undefined;
        size?: {
            width: number;
            height: number;
        } | undefined;
        title?: string | undefined;
        order?: number | undefined;
        collapsed?: boolean | undefined;
        mode?: "default" | "bypass" | "mute" | undefined;
    }>;
    links: readonly {
        id: string;
        source: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        target: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        type?: string | undefined;
    }[];
    metadata?: {
        custom?: Record<string, unknown> | undefined;
        description?: string | undefined;
        tags?: readonly string[] | undefined;
        title?: string | undefined;
        author?: {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        } | {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }[] | undefined;
        createdAt?: string | undefined;
        updatedAt?: string | undefined;
        version?: `${number}.${number}.${number}${string}` | undefined;
        license?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    config?: {
        custom?: Record<string, unknown> | undefined;
        view?: {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        } | undefined;
        selectedNodes?: readonly string[] | undefined;
        focusedNode?: string | undefined;
        showGrid?: boolean | undefined;
        animateLinks?: boolean | undefined;
        linkCurvature?: number | undefined;
        snapToGrid?: {
            size: number;
            enabled: boolean;
        } | undefined;
    } | undefined;
    groups?: readonly {
        id: string;
        title: string;
        bounds: {
            x: number;
            y: number;
            width: number;
            height: number;
        };
        color?: string | undefined;
        properties?: Record<string, unknown> | undefined;
        fontSize?: number | undefined;
        nodeIds?: readonly string[] | undefined;
        locked?: boolean | undefined;
    }[] | undefined;
    reroutes?: readonly {
        id: string;
        position: {
            x: number;
            y: number;
        };
        linkId: string;
        type?: string | undefined;
    }[] | undefined;
    requiredPlugins?: readonly string[] | undefined;
    extra?: Record<string, unknown> | undefined;
}, {
    version: `${number}.${number}.${number}${string}`;
    nodes: Record<string, {
        type: string;
        widgets: Record<string, unknown>;
        id: string;
        position: {
            x: number;
            y: number;
        };
        disabled?: boolean | undefined;
        properties?: Record<string, unknown> | undefined;
        size?: {
            width: number;
            height: number;
        } | undefined;
        title?: string | undefined;
        order?: number | undefined;
        collapsed?: boolean | undefined;
        mode?: "default" | "bypass" | "mute" | undefined;
    }>;
    links: readonly {
        id: string;
        source: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        target: {
            nodeId: string;
            slot: string;
            slotIndex: number;
        };
        type?: string | undefined;
    }[];
    metadata?: {
        custom?: Record<string, unknown> | undefined;
        description?: string | undefined;
        tags?: readonly string[] | undefined;
        title?: string | undefined;
        author?: {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        } | {
            name: string;
            email?: string | undefined;
            url?: string | undefined;
        }[] | undefined;
        createdAt?: string | undefined;
        updatedAt?: string | undefined;
        version?: `${number}.${number}.${number}${string}` | undefined;
        license?: string | undefined;
        thumbnail?: string | undefined;
    } | undefined;
    config?: {
        custom?: Record<string, unknown> | undefined;
        view?: {
            offset: {
                x: number;
                y: number;
            };
            scale: number;
        } | undefined;
        selectedNodes?: readonly string[] | undefined;
        focusedNode?: string | undefined;
        showGrid?: boolean | undefined;
        animateLinks?: boolean | undefined;
        linkCurvature?: number | undefined;
        snapToGrid?: {
            size: number;
            enabled: boolean;
        } | undefined;
    } | undefined;
    groups?: readonly {
        id: string;
        title: string;
        bounds: {
            x: number;
            y: number;
            width: number;
            height: number;
        };
        color?: string | undefined;
        properties?: Record<string, unknown> | undefined;
        fontSize?: number | undefined;
        nodeIds?: readonly string[] | undefined;
        locked?: boolean | undefined;
    }[] | undefined;
    reroutes?: readonly {
        id: string;
        position: {
            x: number;
            y: number;
        };
        linkId: string;
        type?: string | undefined;
    }[] | undefined;
    requiredPlugins?: readonly string[] | undefined;
    extra?: Record<string, unknown> | undefined;
}>;
/**
 * Validate a workflow against the schema.
 */
export declare function validateWorkflow(data: unknown): z.SafeParseReturnType<unknown, z.infer<typeof WorkflowZodSchema>>;
/**
 * Validate a node definition against the schema.
 */
export declare function validateNodeDefinition(data: unknown): z.SafeParseReturnType<unknown, z.infer<typeof NodeDefinitionSchema>>;
/**
 * Parse and validate a workflow, throwing on error.
 */
export declare function parseWorkflow(data: unknown): z.infer<typeof WorkflowZodSchema>;
/**
 * Parse and validate a node definition, throwing on error.
 */
export declare function parseNodeDefinition(data: unknown): z.infer<typeof NodeDefinitionSchema>;
export {};
//# sourceMappingURL=index.d.ts.map