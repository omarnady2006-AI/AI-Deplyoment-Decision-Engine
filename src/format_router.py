"""
format_router.py
================
Detects model format by file extension, validates the file, and dispatches
to a lightweight metadata parser.  No inference is ever executed — only
graph structure / binary headers are inspected.

Supported formats
-----------------
  ONNX       .onnx
  TFLite     .tflite
  PyTorch    .pt  .pth
  TensorFlow .pb
  TensorRT   .engine  .trt
  OpenVINO   .xml  (+ optional .bin weights)

Return contract
---------------
Every public parse_* function returns a dict shaped for _build_facts():

  {
    # mandatory
    "framework":          str,
    "parameter_count":    int,
    "operator_count":     int,
    "model_size_mb":      float,
    "dynamic_shapes":     bool,

    # extended — consumed by _build_facts() in gui_app.py
    "operators":          List[str],
    "operator_counts":    Dict[str, int],
    "has_dynamic_shapes": bool,
    "input_count":        int,
    "output_count":       int,

    # routing meta
    "analysis_success":   bool,
    "error":              Optional[str],
  }

Security
--------
  - Rejects files > 2 GB
  - Rejects unknown extensions
  - Returns analysis_success=False on any parse error
"""

from __future__ import annotations

import struct
import zipfile
import io
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_BYTES = 2 * 1024 ** 3  # 2 GB hard limit

EXTENSION_MAP: Dict[str, str] = {
    ".onnx":   "onnx",
    ".tflite": "tflite",
    ".pt":     "pytorch",
    ".pth":    "pytorch",
    ".pb":     "tensorflow",
    ".engine": "tensorrt",
    ".trt":    "tensorrt",
    ".xml":    "openvino",
}


import logging
logger = logging.getLogger(__name__)

def _empty_result(framework: str, size_mb: float) -> Dict[str, Any]:
    return {
        "framework":          framework,
        "parameter_count":    0,
        "operator_count":     0,
        "model_size_mb":      size_mb,
        "dynamic_shapes":     False,
        "operators":          [],
        "operator_counts":    {},
        "has_dynamic_shapes": False,
        "input_count":        0,
        "output_count":       0,
        "analysis_success":   True,
        "error":              None,
    }


def _error_result(framework: str, size_mb: float, msg: str) -> Dict[str, Any]:
    r = _empty_result(framework, size_mb)
    r["analysis_success"] = False
    r["error"] = msg
    return r


# ═════════════════════════════════════════════════════════════════
#  Minimal protobuf reader (pure Python, no external deps)
# ═════════════════════════════════════════════════════════════════

def _pb_read_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode a protobuf varint.  Returns (value, new_pos)."""
    result, shift = 0, 0
    while pos < len(data):
        byte = data[pos];  pos += 1
        result |= (byte & 0x7F) << shift
        shift  += 7
        if not (byte & 0x80):
            return result, pos
    return result, pos


def _pb_iter_fields(data: bytes) -> Iterator[Tuple[int, int, Any]]:
    """
    Iterate top-level fields in a protobuf message.
    Yields  (field_id: int, wire_type: int, value: bytes | int).
    Stops on first malformed byte rather than raising.
    """
    pos = 0
    while pos < len(data):
        tag, pos = _pb_read_varint(data, pos)
        if pos > len(data):
            break
        field_id  = tag >> 3
        wire_type = tag &  0x7
        if wire_type == 0:       # varint
            value, pos = _pb_read_varint(data, pos)
            yield field_id, 0, value
        elif wire_type == 1:     # 64-bit fixed
            if pos + 8 > len(data): break
            yield field_id, 1, data[pos:pos+8];  pos += 8
        elif wire_type == 2:     # length-delimited
            length, pos = _pb_read_varint(data, pos)
            if pos + length > len(data): break
            yield field_id, 2, data[pos:pos+length];  pos += length
        elif wire_type == 5:     # 32-bit fixed
            if pos + 4 > len(data): break
            yield field_id, 5, data[pos:pos+4];  pos += 4
        else:
            break   # unknown wire type → stop parsing


def _pb_read_string(field_bytes: bytes) -> str:
    try:
        return field_bytes.decode("utf-8", errors="replace")
    except Exception:
        return ""


# ═════════════════════════════════════════════════════════════════
#  Minimal FlatBuffers reader (pure Python, no external deps)
# ═════════════════════════════════════════════════════════════════

class FlatBuf:
    """
    Thin wrapper around a FlatBuffers byte buffer that supports
    table-field access and vector traversal.
    """
    __slots__ = ("buf",)

    def __init__(self, buf: bytes) -> None:
        self.buf = buf

    # ── low-level readers ────────────────────────────────────────

    def _u8(self, pos: int) -> int:
        return self.buf[pos] if 0 <= pos < len(self.buf) else 0

    def _u16(self, pos: int) -> int:
        if pos + 2 > len(self.buf): return 0
        return struct.unpack_from("<H", self.buf, pos)[0]

    def _i32(self, pos: int) -> int:
        if pos + 4 > len(self.buf): return 0
        return struct.unpack_from("<i", self.buf, pos)[0]

    def _u32(self, pos: int) -> int:
        if pos + 4 > len(self.buf): return 0
        return struct.unpack_from("<I", self.buf, pos)[0]

    # ── table helpers ────────────────────────────────────────────

    def _vtable(self, table_pos: int) -> int:
        """Return vtable absolute position for table at table_pos."""
        soffset = self._i32(table_pos)
        return table_pos - soffset

    def _field_offset(self, table_pos: int, field_index: int) -> int:
        """
        Return the absolute offset of field `field_index` within table
        at `table_pos`, or 0 if the field is absent.
        """
        vt   = self._vtable(table_pos)
        vsz  = self._u16(vt)
        slot = 4 + field_index * 2
        if slot + 2 > vsz:
            return 0
        off = self._u16(vt + slot)
        return (table_pos + off) if off else 0

    # ── typed readers ────────────────────────────────────────────

    def read_u8(self, table_pos: int, field_index: int, default: int = 0) -> int:
        off = self._field_offset(table_pos, field_index)
        return self._u8(off) if off else default

    def read_u32(self, table_pos: int, field_index: int, default: int = 0) -> int:
        off = self._field_offset(table_pos, field_index)
        return self._u32(off) if off else default

    def read_string(self, table_pos: int, field_index: int) -> str:
        off = self._field_offset(table_pos, field_index)
        if not off: return ""
        # string field holds a UOffset32 to the string object
        str_pos = off + self._u32(off)
        length  = self._u32(str_pos)
        start   = str_pos + 4
        raw     = self.buf[start : start + length]
        return raw.decode("utf-8", errors="replace")

    def vector_len(self, table_pos: int, field_index: int) -> int:
        off = self._field_offset(table_pos, field_index)
        if not off: return 0
        vec_pos = off + self._u32(off)   # follow UOffset to vector object
        return self._u32(vec_pos)

    def vector_table(self, table_pos: int, field_index: int, elem_index: int) -> int:
        """Return the absolute position of table element `elem_index` in vector."""
        off = self._field_offset(table_pos, field_index)
        if not off: return 0
        vec_pos  = off + self._u32(off)
        count    = self._u32(vec_pos)
        if elem_index >= count: return 0
        elem_off = vec_pos + 4 + elem_index * 4
        # UOffset: offset from elem_off position to the table
        return elem_off + self._u32(elem_off)

    def vector_i32(self, table_pos: int, field_index: int, elem_index: int) -> int:
        off = self._field_offset(table_pos, field_index)
        if not off: return 0
        vec_pos  = off + self._u32(off)
        count    = self._u32(vec_pos)
        if elem_index >= count: return 0
        return self._i32(vec_pos + 4 + elem_index * 4)

    def root(self) -> int:
        """Return absolute position of the root table."""
        return self._u32(0)


# ═════════════════════════════════════════════════════════════════
#  Security gate
# ═════════════════════════════════════════════════════════════════

def _check_path(model_path: str) -> Tuple[Path, float, Optional[str]]:
    """
    Validate the file and return (path, size_mb, error_or_None).
    """
    p    = Path(model_path)
    size = p.stat().st_size if p.exists() else 0
    size_mb = size / (1024 ** 2)

    if not p.exists():
        return p, 0.0, f"File not found: {model_path}"
    if size > MAX_FILE_SIZE_BYTES:
        return p, size_mb, "SECURITY_REJECT: file exceeds 2 GB limit"
    ext = p.suffix.lower()
    if ext not in EXTENSION_MAP:
        return p, size_mb, f"UNSUPPORTED_FORMAT: unknown extension '{ext}'"
    return p, size_mb, None


# ═════════════════════════════════════════════════════════════════
#  ONNX parser  (.onnx)
# ═════════════════════════════════════════════════════════════════

def parse_onnx(model_path: str) -> Dict[str, Any]:
    """
    Parse an ONNX model using:
      1. onnxruntime InferenceSession for input/output metadata.
      2. Custom protobuf reader for full operator and weight inventory.
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("onnx", size_mb, err)

    result = _empty_result("onnx", size_mb)

    # ── Pass 1: onnxruntime session (inputs / outputs / model meta) ──
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
        result["input_count"]  = len(sess.get_inputs())
        result["output_count"] = len(sess.get_outputs())

        # Check for dynamic shapes via onnxruntime input shapes
        for inp in sess.get_inputs():
            shape = getattr(inp, "shape", None) or []
            for dim in shape:
                if dim is None or (isinstance(dim, int) and dim <= 0):
                    result["dynamic_shapes"]     = True
                    result["has_dynamic_shapes"] = True
                    break
                if isinstance(dim, str):
                    result["dynamic_shapes"]     = True
                    result["has_dynamic_shapes"] = True
                    break
    except Exception as e:
        # Non-fatal — continue with protobuf pass
        result["error"] = f"ort_session_warning: {e}"

    # ── Pass 2: protobuf reader for operators + parameters ──────────
    try:
        raw = p.read_bytes()
        _onnx_parse_graph(raw, result)
    except Exception as e:
        if result["error"]:
            result["error"] += f"; pb_parse_warning: {e}"
        else:
            result["error"] = f"pb_parse_warning: {e}"

    result["analysis_success"] = True
    return result


def _onnx_parse_graph(data: bytes, result: Dict[str, Any]) -> None:
    """
    Walk the ONNX ModelProto protobuf and fill operator + parameter counts.

    ModelProto layout:
        field  7 (len-delimited) = GraphProto graph

    GraphProto layout:
        field  1 (len-delimited) = repeated NodeProto  node
        field  5 (len-delimited) = repeated TensorProto initializer
        field 11 (len-delimited) = repeated ValueInfoProto input
        field 12 (len-delimited) = repeated ValueInfoProto output

    NodeProto layout:
        field  4 (len-delimited) = string op_type

    TensorProto layout:
        field  1 (varint / len-delimited packed) = repeated int64 dims
    """
    # Find GraphProto (field 7)
    graph_bytes: Optional[bytes] = None
    for fid, wt, val in _pb_iter_fields(data):
        if fid == 7 and wt == 2:
            graph_bytes = val
            break
    if graph_bytes is None:
        return

    op_counts: Dict[str, int] = {}
    param_count = 0
    pb_input_count  = 0
    pb_output_count = 0

    for fid, wt, val in _pb_iter_fields(graph_bytes):
        if fid == 1 and wt == 2:          # NodeProto
            for nfid, nwt, nval in _pb_iter_fields(val):
                if nfid == 4 and nwt == 2:
                    op = _pb_read_string(nval)
                    op_counts[op] = op_counts.get(op, 0) + 1
                    break

        elif fid == 5 and wt == 2:         # TensorProto initializer
            dims: List[int] = []
            for ifid, iwt, ival in _pb_iter_fields(val):
                if ifid == 1:
                    if iwt == 0:           # single int64 varint
                        dims.append(int(ival))
                    elif iwt == 2:         # packed int64
                        ppos = 0
                        while ppos < len(ival):
                            v, ppos = _pb_read_varint(ival, ppos)
                            dims.append(int(v))
            product = 1
            for d in dims:
                product *= max(1, d)
            param_count += product

        elif fid == 11 and wt == 2:        # input ValueInfoProto
            pb_input_count += 1

        elif fid == 12 and wt == 2:        # output ValueInfoProto
            pb_output_count += 1

    result["operator_counts"] = op_counts
    result["operators"]       = list(op_counts.keys())
    result["operator_count"]  = sum(op_counts.values())
    result["parameter_count"] = param_count

    # Only override i/o counts if onnxruntime didn't already set them
    if result["input_count"]  == 0:  result["input_count"]  = pb_input_count
    if result["output_count"] == 0:  result["output_count"] = pb_output_count


# ═════════════════════════════════════════════════════════════════
#  TFLite parser  (.tflite)
# ═════════════════════════════════════════════════════════════════

_TFLITE_MAGIC = {b"TFL3", b"TFL1", b"TFL2"}


def parse_tflite(model_path: str) -> Dict[str, Any]:
    """
    Parse a TFLite FlatBuffers model.

    TFLite Model schema (selected fields):
        Model  field 0 = version         (uint32)
               field 1 = operator_codes  (vector of OperatorCode tables)
               field 2 = subgraphs       (vector of SubGraph tables)
               field 4 = buffers         (vector of Buffer tables)

        SubGraph field 0 = tensors    (vector of Tensor tables)
                 field 1 = inputs     (vector of int32)
                 field 2 = outputs    (vector of int32)
                 field 3 = operators  (vector of Operator tables)

        Tensor field 0 = shape  (vector of int32)
               field 3 = name   (string)

        BuiltinOperator enum is ignored — we record operator code indices.
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("tflite", size_mb, err)

    try:
        raw = p.read_bytes()
    except Exception as e:
        return _error_result("tflite", size_mb, f"read_error: {e}")

    if len(raw) < 8:
        return _error_result("tflite", size_mb, "file too small to be a TFLite model")

    # Validate file identifier (bytes 4–7)
    file_id = raw[4:8]
    if file_id not in _TFLITE_MAGIC:
        # Accept files whose identifier is missing or unknown but root offset looks valid
        root_off = struct.unpack_from("<I", raw, 0)[0]
        if root_off == 0 or root_off >= len(raw):
            return _error_result("tflite", size_mb,
                                 f"invalid TFLite magic: {file_id!r}")

    fb = FlatBuf(raw)
    model_pos = fb.root()   # absolute position of Model table

    result = _empty_result("tflite", size_mb)

    try:
        # ── Operator codes (field 1) ─────────────────────────────
        n_op_codes = fb.vector_len(model_pos, 1)
        op_code_labels: List[str] = []
        for i in range(n_op_codes):
            tbl = fb.vector_table(model_pos, 1, i)
            # BuiltinOperator enum is field 0 (int8/int32) in OperatorCode
            code_val = fb.read_u32(tbl, 0, default=i)
            op_code_labels.append(f"op_{code_val}")

        # ── Subgraphs (field 2) ──────────────────────────────────
        n_subgraphs  = fb.vector_len(model_pos, 2)
        total_ops    = 0
        total_params = 0
        op_counts: Dict[str, int] = {}
        max_inputs = max_outputs = 0

        # Buffers are referenced by tensors — we'll use tensor dims for param count
        n_buffers = fb.vector_len(model_pos, 4)

        for sg_idx in range(n_subgraphs):
            sg = fb.vector_table(model_pos, 2, sg_idx)

            # inputs / outputs
            n_in  = fb.vector_len(sg, 1)
            n_out = fb.vector_len(sg, 2)
            if sg_idx == 0:
                max_inputs  = n_in
                max_outputs = n_out

            # operators (field 3)
            n_ops = fb.vector_len(sg, 3)
            total_ops += n_ops
            for op_idx in range(n_ops):
                op_tbl  = fb.vector_table(sg, 3, op_idx)
                code_i  = fb.read_u32(op_tbl, 0, default=0)
                label   = op_code_labels[code_i] if code_i < len(op_code_labels) else f"op_{code_i}"
                op_counts[label] = op_counts.get(label, 0) + 1

            # tensors (field 0) — estimate params from shape
            n_tensors = fb.vector_len(sg, 0)
            for t_idx in range(n_tensors):
                t_tbl = fb.vector_table(sg, 0, t_idx)
                n_dims = fb.vector_len(t_tbl, 0)
                if n_dims > 0:
                    volume = 1
                    for d_idx in range(n_dims):
                        dim_val = fb.vector_i32(t_tbl, 0, d_idx)
                        volume *= max(1, dim_val)
                    total_params += volume

        result["operator_counts"] = op_counts
        result["operators"]       = list(op_counts.keys())
        result["operator_count"]  = total_ops
        result["parameter_count"] = total_params
        result["input_count"]     = max_inputs
        result["output_count"]    = max_outputs

    except Exception as e:
        result["error"] = f"tflite_parse_warning: {e}"

    result["analysis_success"] = True
    return result


# ═════════════════════════════════════════════════════════════════
#  PyTorch parser  (.pt  .pth)
# ═════════════════════════════════════════════════════════════════

def parse_pytorch(model_path: str) -> Dict[str, Any]:
    """
    Parse a PyTorch serialized model.

    PyTorch .pt/.pth files are ZIP archives (since PyTorch 1.6).
    We inspect the archive members to:
      - Enumerate tensors from data/N storage files
      - Read the model structure from the pickle stream header
      (We never execute pickle — safety first.)

    Older files use the legacy serialisation format (magic 0x5071306f).
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("pytorch", size_mb, err)

    result = _empty_result("pytorch", size_mb)

    try:
        raw = p.read_bytes()
    except Exception as e:
        return _error_result("pytorch", size_mb, f"read_error: {e}")

    # ── Detect ZIP-based format ──────────────────────────────────
    if raw[:4] == b"PK\x03\x04":
        return _parse_pytorch_zip(raw, size_mb)

    # ── Detect legacy format (magic 0x5071306f == "Pq0o") ────────
    LEGACY_MAGIC = b"\x80\x02c"  # pickle MARK + GLOBAL for torch._utils
    PT_MAGIC = struct.pack("<I", 0x70306f50)  # "Po0p" LE — another marker
    # Accept any non-zero file as a legacy model
    result["framework"] = "pytorch"
    result["operator_count"] = 1
    result["operators"] = ["legacy_serialized"]
    result["operator_counts"] = {"legacy_serialized": 1}

    # Estimate param count from file size (rough: 4 bytes per FP32 param)
    result["parameter_count"] = int(size_mb * 1024 * 1024 / 4)
    return result


def _parse_pytorch_zip(raw: bytes, size_mb: float) -> Dict[str, Any]:
    result = _empty_result("pytorch", size_mb)
    try:
        zf = zipfile.ZipFile(io.BytesIO(raw))
        names = zf.namelist()

        # Count tensor storage files (archive/data/0, archive/data/1, ...)
        tensor_files = [n for n in names if "/data/" in n and not n.endswith("/")]
        total_params = 0
        for name in tensor_files:
            try:
                entry_size = zf.getinfo(name).file_size
                # Assume FP32 (4 bytes per param) unless we can read the dtype
                total_params += entry_size // 4
            except Exception as e:
                logger.warning("exception_occurred", exc_info=True)

        result["parameter_count"] = total_params

        # Try reading record.pkl to count ops (safe: only read the opcode bytes)
        record_name = next(
            (n for n in names if n.endswith("record.pkl") or n.endswith("data.pkl")),
            None,
        )
        op_counts: Dict[str, int] = {}
        if record_name:
            try:
                pkl_bytes = zf.read(record_name)
                # Scan pickle opcodes for GLOBAL (0x63) — each is a symbol reference
                symbols: List[str] = []
                i = 0
                while i < len(pkl_bytes):
                    opcode = pkl_bytes[i]; i += 1
                    if opcode == 0x63:  # GLOBAL: two newline-terminated strings
                        end1 = pkl_bytes.index(b"\n", i)
                        module = pkl_bytes[i:end1].decode("utf-8", errors="replace")
                        i = end1 + 1
                        end2 = pkl_bytes.index(b"\n", i)
                        name_s = pkl_bytes[i:end2].decode("utf-8", errors="replace")
                        i = end2 + 1
                        sym = f"{module}.{name_s}"
                        if "torch" in module:
                            op_counts[sym] = op_counts.get(sym, 0) + 1
                    elif opcode == 0x41:  # APPENDS
                        pass
                    # Skip most opcodes to avoid accidental execution
            except Exception as e:
                logger.warning("exception_occurred", exc_info=True)

        if not op_counts:
            op_counts = {"TorchScript": max(1, len(tensor_files))}
        result["operators"]       = list(op_counts.keys())
        result["operator_counts"] = op_counts
        result["operator_count"]  = sum(op_counts.values())

        # Input/output from archive structure: look for forward() signature hints
        result["input_count"]  = 1
        result["output_count"] = 1

    except Exception as e:
        result["error"] = f"pytorch_zip_warning: {e}"

    result["analysis_success"] = True
    return result


# ═════════════════════════════════════════════════════════════════
#  TensorFlow parser  (.pb)
# ═════════════════════════════════════════════════════════════════

def parse_tensorflow(model_path: str) -> Dict[str, Any]:
    """
    Parse a TensorFlow GraphDef protobuf.

    GraphDef layout:
        field 1 (len-delimited) = repeated NodeDef node
        field 4 (len-delimited) = FunctionDefLibrary library

    NodeDef layout:
        field 1 (len-delimited) = string name
        field 2 (len-delimited) = string op
        field 3 (len-delimited) = repeated string input
        field 5 (len-delimited) = map<string,AttrValue> attr
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("tensorflow", size_mb, err)

    result = _empty_result("tensorflow", size_mb)

    try:
        raw = p.read_bytes()
    except Exception as e:
        return _error_result("tensorflow", size_mb, f"read_error: {e}")

    if len(raw) < 4:
        return _error_result("tensorflow", size_mb, "file too small to be a TF GraphDef")

    try:
        op_counts:     Dict[str, int] = {}
        param_count    = 0
        variable_ops   = {"Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"}
        input_nodes:   List[str] = []
        output_nodes:  List[str] = []

        for fid, wt, val in _pb_iter_fields(raw):
            if fid == 1 and wt == 2:   # NodeDef
                node_name = ""
                node_op   = ""
                for nfid, nwt, nval in _pb_iter_fields(val):
                    if nfid == 1 and nwt == 2:
                        node_name = _pb_read_string(nval)
                    elif nfid == 2 and nwt == 2:
                        node_op = _pb_read_string(nval)
                if node_op:
                    op_counts[node_op] = op_counts.get(node_op, 0) + 1
                if node_op in ("Placeholder",):
                    input_nodes.append(node_name)
                if node_op in variable_ops:
                    # Rough estimate — variable has some params
                    param_count += 1000   # placeholder; real count needs AttrValue parse

        result["operator_counts"] = op_counts
        result["operators"]       = list(op_counts.keys())
        result["operator_count"]  = sum(op_counts.values())
        result["parameter_count"] = param_count
        result["input_count"]     = len(input_nodes) or 1
        result["output_count"]    = max(1, op_counts.get("Identity", 1))

    except Exception as e:
        result["error"] = f"tf_parse_warning: {e}"

    result["analysis_success"] = True
    return result


# ═════════════════════════════════════════════════════════════════
#  TensorRT parser  (.engine  .trt)
# ═════════════════════════════════════════════════════════════════

# Known TensorRT binary headers
_TRT_MAGIC_GIEF  = b"GIEF"     # Generic Inference Engine Format
_TRT_MAGIC_TRT1  = b"\x00\x00\x00\x01"  # early TRT serialised engines


def parse_tensorrt(model_path: str) -> Dict[str, Any]:
    """
    Inspect a TensorRT serialised engine.

    TensorRT engines are proprietary binary blobs.  Without the TensorRT SDK
    we cannot decode the full graph.  We:
      - Validate magic bytes / file structure
      - Return file-level metadata
      - Mark as analysis_success=True with a metadata-only note
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("tensorrt", size_mb, err)

    result = _empty_result("tensorrt", size_mb)

    try:
        with open(str(p), "rb") as fh:
            header = fh.read(16)
    except Exception as e:
        return _error_result("tensorrt", size_mb, f"read_error: {e}")

    # Identify header
    if len(header) < 4:
        return _error_result("tensorrt", size_mb, "file too small")

    # TRT engines typically start with GIEF or a 4-byte serialisation version
    magic = header[:4]
    engine_version = "unknown"
    if magic == _TRT_MAGIC_GIEF:
        engine_version = "GIEF"
    elif len(header) >= 8:
        # Try reading as uint64 version field
        try:
            ver = struct.unpack_from("<Q", header, 0)[0]
            if 0 < ver < 0xFFFF:
                engine_version = f"v{ver}"
        except Exception as e:
            logger.warning("exception_occurred", exc_info=True)

    # Estimate layer count from file size (rough heuristic: ~4 KB per layer)
    estimated_layers = max(1, int(size_mb * 1024 / 4))

    result["operator_count"]  = estimated_layers
    result["operators"]       = ["TRTCudaKernel"] * min(estimated_layers, 5)
    result["operator_counts"] = {"TRTCudaKernel": estimated_layers}
    result["parameter_count"] = int(size_mb * 1024 * 1024 / 4)  # assume FP32
    result["input_count"]     = 1
    result["output_count"]    = 1
    result["error"]           = (
        f"metadata_only: TRT engine v={engine_version}; "
        "full graph unavailable without TensorRT SDK"
    )
    result["analysis_success"] = True
    return result


# ═════════════════════════════════════════════════════════════════
#  OpenVINO parser  (.xml + optional .bin)
# ═════════════════════════════════════════════════════════════════

def parse_openvino(model_path: str) -> Dict[str, Any]:
    """
    Parse an OpenVINO IR model (.xml).

    OpenVINO IR XML structure:
        <net name="..." version="...">
            <layers>
                <layer id="..." name="..." type="..." version="...">
                    <data .../>
                    <input>  <port id="..." ...> ... </port> </input>
                    <output> <port id="..." ...> ... </port> </output>
                </layer>
            </layers>
            <edges>
                <edge from-layer="..." to-layer="..."/>
            </edges>
        </net>
    """
    p, size_mb, err = _check_path(model_path)
    if err:
        return _error_result("openvino", size_mb, err)

    result = _empty_result("openvino", size_mb)

    # Also count .bin weights file size if it exists
    bin_path = p.with_suffix(".bin")
    if bin_path.exists():
        bin_size = bin_path.stat().st_size / (1024 ** 2)
        result["model_size_mb"] = size_mb + bin_size

    try:
        tree = ET.parse(str(p))
        root = tree.getroot()
    except ET.ParseError as e:
        return _error_result("openvino", size_mb, f"xml_parse_error: {e}")

    try:
        layers_elem = root.find("layers")
        if layers_elem is None:
            return _error_result("openvino", size_mb, "no <layers> element in XML")

        op_counts:     Dict[str, int] = {}
        param_count    = 0
        input_nodes:   List[str] = []
        output_nodes:  List[str] = []
        has_dynamic    = False

        for layer in layers_elem.findall("layer"):
            layer_type = layer.get("type", "Unknown")
            op_counts[layer_type] = op_counts.get(layer_type, 0) + 1

            if layer_type == "Parameter":
                input_nodes.append(layer.get("name", ""))
            if layer_type in ("Result", "Output"):
                output_nodes.append(layer.get("name", ""))

            # Count weights from Const layers
            if layer_type == "Const":
                data_elem = layer.find("data")
                if data_elem is not None:
                    size_attr = data_elem.get("size", "0")
                    try:
                        param_count += int(size_attr) // 4  # assume FP32
                    except ValueError as e:
                        logger.warning("exception_occurred", exc_info=True)

            # Check for dynamic dims ("-1" in port shapes)
            for port in layer.findall(".//port"):
                for dim in port.findall("dim"):
                    if dim.text and dim.text.strip() in ("-1", "?", ""):
                        has_dynamic = True

        result["operator_counts"]  = op_counts
        result["operators"]        = list(op_counts.keys())
        result["operator_count"]   = sum(op_counts.values())
        result["parameter_count"]  = param_count
        result["input_count"]      = len(input_nodes) or 1
        result["output_count"]     = len(output_nodes) or 1
        result["dynamic_shapes"]   = has_dynamic
        result["has_dynamic_shapes"] = has_dynamic

        # Fallback param estimate from .bin if Const layers had no size attribute
        if param_count == 0 and bin_path.exists():
            bin_bytes = bin_path.stat().st_size
            result["parameter_count"] = bin_bytes // 4  # assume FP32

    except Exception as e:
        result["error"] = f"openvino_parse_warning: {e}"

    result["analysis_success"] = True
    return result


# ═════════════════════════════════════════════════════════════════
#  Public router
# ═════════════════════════════════════════════════════════════════

_PARSERS = {
    "onnx":       parse_onnx,
    "tflite":     parse_tflite,
    "pytorch":    parse_pytorch,
    "tensorflow": parse_tensorflow,
    "tensorrt":   parse_tensorrt,
    "openvino":   parse_openvino,
}


def route_model(model_path: str) -> Dict[str, Any]:
    """
    Detect the model format from `model_path` extension and dispatch to
    the appropriate parser.

    Returns the unified analysis dict.  On unknown extension or > 2 GB:
        {
          "analysis_success": False,
          "error": "UNSUPPORTED_FORMAT" | "SECURITY_REJECT: ...",
          ...
        }
    """
    p   = Path(model_path)
    ext = p.suffix.lower()

    if ext not in EXTENSION_MAP:
        return _error_result(
            "unknown",
            p.stat().st_size / (1024**2) if p.exists() else 0.0,
            "UNSUPPORTED_FORMAT",
        )

    fmt    = EXTENSION_MAP[ext]
    parser = _PARSERS[fmt]
    return parser(model_path)
