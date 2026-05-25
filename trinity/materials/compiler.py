"""AST to WGSL compiler for Material DSL."""
import textwrap
import ast
import inspect


class MaterialCompiler:
    TYPE_MAP = {float: "f32", int: "i32", bool: "bool", str: "str"}

    def compile(self, material_class: type) -> str:
        """Compile a Material subclass to WGSL surface function body."""
        source = textwrap.dedent(inspect.getsource(material_class.surface))
        tree = ast.parse(source)
        # Walk AST, translate Python expressions to WGSL
        return self._walk(tree)

    def _walk(self, node) -> str:
        # Stub: return placeholder WGSL
        return "// WGSL surface body placeholder"
