use byteorder::{ByteOrder, LittleEndian};
use leb128;

mod name_section;

#[derive(Debug, Clone)]
pub struct FuncType {
    pub params: Vec<u8>,
    pub results: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct FuncExport {
    pub name: String,
    pub func: Func,
}

#[derive(PartialEq, Clone)]
pub enum ExportType {
    Func,
    Mem,
    Global,
}

#[derive(PartialEq, Clone)]
pub enum ImportType {
    Func,
}

#[derive(Debug, Clone)]
pub struct FuncCode {
    pub opcode: u8,
    pub immediates: Vec<Imm>,
    pub return_type: Option<u8>,
}

#[derive(Debug, Clone)]
pub enum Imm {
    // Some intruction have reserved bytes, which must be a single 0 byte.
    // Using a long leb128 encoding of 0 is not valid
    RESERVED,

    I64(i64),
    I32(i32),
    F64(f64),
}
impl From<i64> for Imm {
    #[inline]
    fn from(v: i64) -> Self {
        Imm::I64(v)
    }
}

impl FuncCode {
    pub fn new0(opcode: u8) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![],
            return_type: None,
        }
    }

    pub fn new_control(opcode: u8, rt: u8) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![],
            return_type: Some(rt),
        }
    }

    pub fn new1(opcode: u8, imm: Imm) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![imm],
            return_type: None,
        }
    }

    pub fn new2(opcode: u8, imm1: Imm, imm2: Imm) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![imm1, imm2],
            return_type: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Element {
    pub table: u32,
    pub offset: u32,
    pub funcs: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Table {
    pub elemtype: TableElemType,
    pub limits: (u32, u32), // (min, max)
}

#[derive(Debug, Clone, Copy)]
pub enum TableElemType {
    Funcref = 0x70,
}

#[derive(Debug, Clone)]
pub struct Func {
    pub sig: FuncType,
    pub locals: Vec<FuncLocal>,
    pub code: Vec<FuncCode>,
}

type Export = (String, usize, ExportType);
type Import = (String, String, ImportType, usize);
type Global = (u8, u8, u32); // (type, mutability, init)
type FuncLocal = (usize, u8);

pub const HEADER_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6D];
pub const HEADER_VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

pub const CUSTOM_SECTION: u8 = 0;
pub const TYPE_SECTION: u8 = 1;
pub const IMPORT_SECTION: u8 = 2;
pub const FUNCTION_SECTION: u8 = 3;
pub const TABLE_SECTION: u8 = 4;
pub const MEMORY_SECTION: u8 = 5;
pub const GLOBAL_SECTION: u8 = 6;
pub const EXPORT_SECTION: u8 = 7;
pub const START_SECTION: u8 = 8;
pub const ELEMENT_SECTION: u8 = 9;
pub const CODE_SECTION: u8 = 10;
pub const DATA_SECTION: u8 = 11;

pub const UNREACHABLE: u8 = 0x00;

pub const NONE: u8 = 0x40;
pub const I32: u8 = 0x7F;
pub const I64: u8 = 0x7E;
pub const F32: u8 = 0x7D;
pub const F64: u8 = 0x7C;

pub const I32_CONST: u8 = 0x41;
pub const I64_CONST: u8 = 0x42;
pub const F32_CONST: u8 = 0x43;
pub const F64_CONST: u8 = 0x44;

pub const I32_EQZ: u8 = 0x45;
pub const I32_EQ: u8 = 0x46;
pub const I32_NE: u8 = 0x47;
pub const I32_LT_S: u8 = 0x48;
pub const I32_LT_U: u8 = 0x49;
pub const I32_GT_S: u8 = 0x4A;
pub const I32_GT_U: u8 = 0x4B;
pub const I32_LE_S: u8 = 0x4C;
pub const I32_LE_U: u8 = 0x4D;
pub const I32_GE_S: u8 = 0x4E;
pub const I32_ADD: u8 = 0x6A;
pub const I32_SUB: u8 = 0x6B;
pub const I32_MUL: u8 = 0x6C;
pub const I32_DIV_S: u8 = 0x6D;
pub const I32_REM_S: u8 = 0x6F;
pub const F32_DIV: u8 = 0x95;
pub const F64_ADD: u8 = 0xA0;
pub const F64_SUB: u8 = 0xA1;
pub const F64_MUL: u8 = 0xA2;
pub const F64_DIV: u8 = 0xA3;
pub const F64_LE: u8 = 0x65;
pub const F64_GE: u8 = 0x66;
pub const I32_TRUNC_F32_S: u8 = 0xA8;
pub const I32_TRUNC_F64_S: u8 = 0xAA;
pub const F64_CONVERT_I32_S: u8 = 0xB7;

pub const DROP: u8 = 0x1A;

pub const LOCAL_GET: u8 = 0x20;
pub const LOCAL_SET: u8 = 0x21;
pub const LOCAL_TEE: u8 = 0x22;
pub const GLOBAL_GET: u8 = 0x23;
pub const GLOBAL_SET: u8 = 0x24;

pub const I32_LOAD: u8 = 0x28;
pub const I32_LOAD8_U: u8 = 0x2D;
pub const I32_STORE: u8 = 0x36;
pub const I32_STORE8: u8 = 0x3A;

pub const BLOCK: u8 = 0x02;
pub const LOOP: u8 = 0x03;
pub const IF: u8 = 0x04;
pub const ELSE: u8 = 0x05;
pub const BR: u8 = 0x0C;
pub const BR_IF: u8 = 0x0D;

pub const END: u8 = 0x0B;
pub const RETURN: u8 = 0x0F;
pub const CALL: u8 = 0x10;
pub const CALL_INDIRECT: u8 = 0x11;

pub struct WasmCodeGen {
    funcs: Vec<(usize, Func)>,
    types: Vec<FuncType>,
    exports: Vec<Export>,
    elements: Vec<Element>,
    tables: Vec<Table>,
    memories: Vec<(u32, u32)>,
    data: Vec<(u32, Vec<u8>)>, // (offset, bytes)
    imports: Vec<Import>,
    globals: Vec<Global>,
    /// Custom name section for funcs
    func_names: Vec<name_section::Naming>,
}

fn write_name(bytes: &mut Vec<u8>, name: String) {
    write_unsigned_leb128(bytes, name.len() as u64);
    // TODO(sven): is this UTF8?
    bytes.extend(name.into_bytes());
}

fn write_unsigned_leb128(bytes: &mut Vec<u8>, n: u64) {
    leb128::write::unsigned(bytes, n).expect("could not write LEB128");
}

fn write_signed_leb128(bytes: &mut Vec<u8>, n: i64) {
    leb128::write::signed(bytes, n).expect("could not write LEB128");
}

fn write_float(bytes: &mut Vec<u8>, n: f64) {
    let mut b = [0; 8];
    LittleEndian::write_f64(&mut b, n);
    bytes.extend(b.iter())
}

fn write_unsigned_leb128_at_offset(bytes: &mut Vec<u8>, offset: usize, n: usize) {
    // remove placeholder
    bytes.remove(offset);

    let mut buffer = vec![];

    leb128::write::unsigned(&mut buffer, n as u64).expect("could not write LEB128");

    let mut i = 0;
    for byte in buffer {
        bytes.insert(offset + i, byte);
        i += 1;
    }
}

fn write_vec_len<T>(bytes: &mut Vec<u8>, vec: &Vec<T>) {
    write_unsigned_leb128(bytes, vec.len() as u64);
}

fn write_type_section(bytes: &mut Vec<u8>, types: &Vec<FuncType>) {
    write_vec_len(bytes, types); // vec length

    for functype in types {
        bytes.push(0x60); // functype

        write_vec_len(bytes, &functype.params); // vec length
        for b in &functype.params {
            bytes.push(*b);
        }

        write_vec_len(bytes, &functype.results); // vec length
        for b in &functype.results {
            bytes.push(*b);
        }
    }
}

fn write_func_section(bytes: &mut Vec<u8>, funcs: &Vec<(usize, Func)>) {
    write_vec_len(bytes, funcs); // vec length

    for func in funcs {
        write_unsigned_leb128(bytes, func.0 as u64);
    }
}

fn write_element_section(bytes: &mut Vec<u8>, elements: &Vec<Element>) {
    write_vec_len(bytes, elements); // vec length

    for element in elements {
        write_unsigned_leb128(bytes, element.table as u64);

        let offset_expr = vec![FuncCode::new1(I32_CONST, Imm::I64(element.offset as i64))];
        write_code_expr(bytes, &offset_expr);

        write_vec_len(bytes, &element.funcs); // vec length
        for func in &element.funcs {
            write_unsigned_leb128(bytes, func.clone() as u64);
        }
    }
}

fn write_table_section(bytes: &mut Vec<u8>, tables: &Vec<Table>) {
    write_vec_len(bytes, tables); // vec length

    for table in tables {
        bytes.push(table.elemtype as u8);

        let (min, max) = table.limits;
        bytes.push(0x01);
        write_unsigned_leb128(bytes, min as u64);
        write_unsigned_leb128(bytes, max as u64);
    }
}

fn write_imports_section(bytes: &mut Vec<u8>, imports: &Vec<Import>) {
    write_vec_len(bytes, imports); // vec length

    for import in imports {
        write_name(bytes, import.0.clone());
        write_name(bytes, import.1.clone());

        match import.2 {
            ImportType::Func => bytes.push(0x0),
        }

        write_unsigned_leb128(bytes, import.3 as u64);
    }
}

fn write_code_local(bytes: &mut Vec<u8>, locals: &Vec<FuncLocal>) {
    write_vec_len(bytes, locals); // vec length

    for local in locals {
        write_unsigned_leb128(bytes, local.0 as u64);
        bytes.push(local.1);
    }
}

fn write_code_expr(bytes: &mut Vec<u8>, codes: &Vec<FuncCode>) {
    for code in codes {
        bytes.push(code.opcode);
        if let Some(rt) = code.return_type {
            write_unsigned_leb128(bytes, rt as u64);
        }
        for imm in &code.immediates {
            match imm {
                Imm::I64(n) => write_signed_leb128(bytes, *n),
                Imm::I32(n) => write_signed_leb128(bytes, *n as i64),
                Imm::F64(f) => write_float(bytes, *f),
                Imm::RESERVED => bytes.push(0x0),
            };
        }
    }

    bytes.push(END); // end
}

fn write_code_section(bytes: &mut Vec<u8>, funcs: &Vec<(usize, Func)>) {
    write_vec_len(bytes, funcs); // vec length

    for func in funcs {
        let before_offset = bytes.len();
        bytes.push(0x0); // func size

        write_code_local(bytes, &func.1.locals);
        write_code_expr(bytes, &func.1.code);

        let after_offset = bytes.len();

        // func size fixup
        let func_len = after_offset - before_offset - 1;

        write_unsigned_leb128_at_offset(bytes, before_offset, func_len);
    }
}

fn write_data_section(bytes: &mut Vec<u8>, datum: &Vec<(u32, Vec<u8>)>) {
    write_vec_len(bytes, datum); // vec length

    for data in datum {
        bytes.push(0x0); // memidx

        bytes.push(I32_CONST);
        write_signed_leb128(bytes, data.0 as i64); // offset
        bytes.push(END);

        write_vec_len(bytes, &data.1); // vec length
        for b in &data.1 {
            bytes.push(*b);
        }
    }
}

fn write_custom_name_section(bytes: &mut Vec<u8>, names: &Vec<name_section::Naming>) {
    write_name(bytes, "name".to_string());

    // Assigns names to functions
    name_section::write_var_uint7(1, bytes);

    // need to store the current office to fixup later
    let name_payload_len_offset = bytes.len();
    // push 0 for now
    bytes.push(0);

    name_section::write_var_uint32(names.len() as u32, bytes);

    for name in names {
        name_section::write_var_uint32(name.index as u32, bytes);
        name_section::write_var_uint32(name.name.len() as u32, bytes);
        bytes.extend_from_slice(&name.name.as_bytes())
    }

    let after_offset = bytes.len();
    let section_len = after_offset - name_payload_len_offset - 1;

    // fixup section len
    write_unsigned_leb128_at_offset(bytes, name_payload_len_offset, section_len);
}

fn write_export_section(bytes: &mut Vec<u8>, exports: &Vec<Export>) {
    write_vec_len(bytes, exports); // vec length

    for export in exports {
        let (name, idx, export_type) = export;

        write_name(bytes, name.clone());

        match *export_type {
            ExportType::Func => bytes.push(0x0),
            ExportType::Mem => bytes.push(0x2),
            ExportType::Global => bytes.push(0x3),
        }
        write_unsigned_leb128(bytes, *idx as u64);
    }
}

fn write_memory_section(bytes: &mut Vec<u8>, memories: &Vec<(u32, u32)>) {
    write_vec_len(bytes, memories); // vec length

    for mem in memories {
        let (min, max) = mem;
        bytes.push(0x01);
        write_unsigned_leb128(bytes, *min as u64);
        write_unsigned_leb128(bytes, *max as u64);
    }
}

fn write_global_section(bytes: &mut Vec<u8>, globals: &Vec<Global>) {
    write_vec_len(bytes, globals); // vec length

    for data in globals {
        let (t, mutability, init) = data;
        bytes.push(*t);
        bytes.push(*mutability);

        let expr = vec![FuncCode::new1(I32_CONST, Imm::I64(*init as i64))];
        write_code_expr(bytes, &expr);
    }
}

macro_rules! write_section {
    ($b: expr, $o:expr, $id:expr, $write_fn:expr) => {
        if $o.len() > 0 {
            $b.push($id); // section id

            let before_offset = $b.len();
            $b.push(0x0); // section bytes

            $write_fn(&mut $b, &$o);

            let after_offset = $b.len();

            // section fixup
            let section_len = after_offset - before_offset - 1;

            // section length - fixup
            write_unsigned_leb128_at_offset(&mut $b, before_offset, section_len);
        }
    };
}

impl WasmCodeGen {
    pub fn new() -> WasmCodeGen {
        WasmCodeGen {
            types: vec![],
            funcs: vec![],
            tables: vec![],
            exports: vec![],
            elements: vec![],
            memories: vec![],
            data: vec![],
            imports: vec![],
            globals: vec![],
            func_names: vec![],
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];

        bytes.extend(&HEADER_MAGIC);
        bytes.extend(&HEADER_VERSION);

        write_section!(bytes, self.types, TYPE_SECTION, write_type_section);
        write_section!(bytes, self.imports, IMPORT_SECTION, write_imports_section);
        write_section!(bytes, self.funcs, FUNCTION_SECTION, write_func_section);
        write_section!(bytes, self.tables, TABLE_SECTION, write_table_section);
        write_section!(bytes, self.memories, MEMORY_SECTION, write_memory_section);
        write_section!(bytes, self.globals, GLOBAL_SECTION, write_global_section);
        write_section!(bytes, self.exports, EXPORT_SECTION, write_export_section);
        write_section!(bytes, self.elements, ELEMENT_SECTION, write_element_section);
        write_section!(bytes, self.funcs, CODE_SECTION, write_code_section);
        write_section!(bytes, self.data, DATA_SECTION, write_data_section);
        write_section!(
            bytes,
            self.func_names,
            CUSTOM_SECTION,
            write_custom_name_section
        );

        bytes
    }

    pub fn set_name(&mut self, idx: usize, name: String) {
        self.func_names.push(name_section::Naming {
            index: idx as u32,
            name,
        });
    }

    pub fn add_type(&mut self, t: FuncType) -> usize {
        let idx = self.types.len();
        self.types.push(t);

        idx
    }

    pub fn add_export(&mut self, name: String, idx: usize, export_type: ExportType) {
        self.exports.push((name, idx, export_type));
    }

    pub fn add_func(&mut self, f: Func) -> usize {
        let funcidx = self.funcs.len() + self.imports.len();

        self.funcs.push((self.types.len(), f.clone()));
        self.add_type(f.sig);

        funcidx
    }

    pub fn replace_code_func(&mut self, idx: usize, code: Vec<FuncCode>) {
        // we don't store the imports in the funcs map and the result of add_func includes it, so remove them.
        let idx = idx - self.imports.len();
        let (type_idx, old_func) = self.funcs[idx].clone();
        let new_func = Func {
            sig: old_func.sig,
            locals: old_func.locals,
            code,
        };
        self.funcs[idx] = (type_idx, new_func);
    }

    pub fn add_table(&mut self, elemtype: TableElemType, min: u32, max: u32) -> usize {
        let idx = self.tables.len();

        self.tables.push(Table {
            elemtype,
            limits: (min, max),
        });

        idx
    }

    pub fn add_element(&mut self, table: u32, offset: u32, funcs: Vec<u32>) -> usize {
        let idx = self.elements.len();

        self.elements.push(Element {
            table,
            offset,
            funcs,
        });

        idx
    }

    pub fn add_memory(&mut self, min: u32, max: u32) -> usize {
        assert!(self.memories.len() == 0);
        self.memories.push((min, max));

        0
    }

    pub fn add_data(&mut self, offset: u32, bytes: Vec<u8>) -> u32 {
        self.data.push((offset, bytes.clone()));
        bytes.len() as u32
    }

    pub fn add_import(
        &mut self,
        module: String,
        name: String,
        import_type: ImportType,
        typeidx: usize,
    ) -> usize {
        let importidex = self.imports.len();
        self.imports.push((module, name, import_type, typeidx));

        importidex
    }

    pub fn add_mutable_global(&mut self, valtype: u8, init: u32) -> usize {
        let idx = self.globals.len();
        self.globals.push((valtype, 0x01, init));

        idx
    }
}
