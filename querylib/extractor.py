"""Extract documentation for any python package."""

import importlib
import inspect
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from docstring_parser import parse
import sys
import os


class DocstringExtractor:
    """
    Efficiently extracts documentation from Python modules with correct path generation.

    Key Optimizations:
    - Concurrent processing
    - Caching
    - Accurate module path tracking

    Examples
    --------
    >>> d = DocstringExtractor("numpy", max_workers=4)
    >>> doc = d.extract_docs()
    >>> d.save_to_json(doc)
    """

    def __init__(self, module_name: str, max_workers: int = None):
        """
        Initialize the extractor with module name and optional concurrency settings.

        Args:
            module_name (str): Name of the module to extract docs from
            max_workers (int, optional): Number of threads for concurrent processing
        """
        self.module_name = module_name
        self.max_workers = max_workers or (os.cpu_count() or 1) * 2
        self.module = self._safe_import_module(module_name)

    @lru_cache(maxsize=128)
    def _safe_import_module(self, name: str):
        """
        Safely import module with caching to prevent repeated imports.

        Args:
            name (str): Module name to import

        Returns:
            module: Imported module
        """
        try:
            return importlib.import_module(name)
        except ImportError:
            print(f"Could not import module: {name}")
            return None

    def _get_clean_module_path(self, obj, parent_module=None):
        """
        Generate an importable module path for a given object.

        Args:
            obj: The object to get the path for.
            parent_module: Optional parent module.

        Returns:
            str: Importable module path.
        """
        try:
            # Get the actual module
            module = inspect.getmodule(obj) or self.module
            module_name = module.__name__ if module else self.module_name

            # Ensure module belongs to target package
            if not module_name.startswith(self.module_name):
                module_name = self.module_name

            # Get qualified name (handles nested classes/methods)
            qualified_name = getattr(obj, "__qualname__", getattr(obj, "__name__", ""))

            # Try resolving the correct parent path
            parts = module_name.split(".")
            for i in range(len(parts), 0, -1):
                candidate_module = ".".join(parts[:i])
                try:
                    imported_mod = importlib.import_module(candidate_module)
                    if hasattr(imported_mod, qualified_name.split(".")[0]):
                        return f"{candidate_module}.{qualified_name}"
                except ImportError:
                    continue  # Try a higher-level module

            return f"{module_name}.{qualified_name}"  # Fallback
        except Exception as e:
            print(f"Error generating path for {obj}: {e}")
            return self.module_name

    def _extract_item_docs(self, item, parent_module=None, parent_path=""):
        """
        Extract documentation for a single item with clean path generation.

        Args:
            item (tuple): Inspection result of a module item
            parent_module (module, optional): Parent module context
            parent_path (str, optional): Parent path context

        Returns:
            dict or None: Structured documentation information
        """
        name, obj = item

        # Skip private/test items
        if (
            name.startswith("_")
            or name.startswith("test")
            or not (inspect.isclass(obj) or inspect.isfunction(obj))
        ):
            return None

        try:
            # Generate clean path
            full_path = self._get_clean_module_path(obj, parent_module)

            # Parse docstring safely
            doc = inspect.getdoc(obj) or ""
            parsed = parse(doc)

            # Create base doc info
            doc_info = {
                "path": full_path,
                "name": name,
                "summary": parsed.short_description or "",
                "type": "class" if inspect.isclass(obj) else "function",
                "params": [
                    {
                        "name": param.arg_name,
                        "type": param.type_name or "",
                        "description": param.description or "",
                    }
                    for param in parsed.params
                ],
                "returns": (
                    {
                        "name": parsed.returns.return_name,
                        "type": parsed.returns.type_name,
                    }
                    if parsed.returns
                    else None
                ),
                "raises": (
                    [
                        {"description": itr.description, "type": itr.type_name}
                        for itr in parsed.raises
                    ]
                    if parsed.raises
                    else None
                ),
                "examples": [example.description or "" for example in parsed.examples],
            }

            # For classes, include method documentation
            if inspect.isclass(obj):
                methods = [
                    self._extract_item_docs((method_name, method), obj, full_path)
                    for method_name, method in inspect.getmembers(
                        obj, predicate=inspect.isfunction
                    )
                    if not method_name.startswith("_")
                    and not method_name.startswith("test")
                ]
                doc_info["methods"] = [method for method in methods if method]

            return doc_info

        except Exception as e:
            print(f"Error processing {name}: {e}")
            return None

    def extract_docs(self, depth: int = sys.maxsize):
        """
        Extract documentation with concurrent processing and iterative module exploration.

        Args:
            depth (int): Maximum exploration depth for modules

        Returns:
            list: Structured documentation
        """
        if not self.module:
            return []

        docs = []
        processed_modules = set()
        module_queue = [(self.module, "", 0)]

        while module_queue:
            current_module, current_path, current_depth = module_queue.pop(0)

            if current_module in processed_modules or current_depth > depth:
                continue

            processed_modules.add(current_module)
            module_docs = []

            try:
                # Use thread pool for concurrent processing
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Process module members
                    futures = {
                        executor.submit(
                            self._extract_item_docs, item, current_module
                        ): item
                        for item in inspect.getmembers(current_module)
                    }

                    for future in as_completed(futures):
                        result = future.result()
                        if result:
                            module_docs.append(result)

                    # Explore submodules iteratively
                    if current_depth < depth:
                        for name, member in inspect.getmembers(current_module):
                            if (
                                inspect.ismodule(member)
                                and member.__name__.startswith(self.module_name)
                                and member not in processed_modules
                            ):
                                module_queue.append((member, name, current_depth + 1))

                    docs.extend(module_docs)

            except Exception as e:
                print(f"Error exploring module {current_path}: {e}")

        return docs

    def save_to_json(self, docs, filename=None, indent=2):
        """
        Save extracted documentation to a JSON file with pretty formatting.

        Args:
            docs (list): Documentation entries to save
            filename (str, optional): Output filename. If None, uses module name.
            indent (int, optional): JSON indentation level

        Returns:
            str: Path to the saved JSON file
        """
        # Use module name if no filename provided
        if filename is None:
            filename = f"{self.module_name}_docs.json"

        # Ensure the filename ends with .json
        if not filename.endswith(".json"):
            filename += ".json"

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

        # Write documentation to file with pretty formatting
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=indent, ensure_ascii=False)

        print(f"Documentation saved to {filename}")
        return filename
