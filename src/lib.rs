use leb128;

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

#[derive(PartialEq)]
pub enum ExportType {
    Func,
    Mem,
    Global
}

#[derive(PartialEq)]
pub enum ImportType {
    Func,
}

#[derive(Debug, Clone)]
pub struct FuncCode {
    pub opcode: u8,
    pub immediates: Vec<u64>,
}

impl FuncCode {
    pub fn new0(opcode: u8) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![]
        }
    }

    pub fn new1(opcode: u8, imm: u64) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![imm]
        }
    }

    pub fn new2(opcode: u8, imm1: u64, imm2: u64) -> FuncCode {
        FuncCode {
            opcode,
            immediates: vec![imm1, imm2]
        }
    }
}

#[derive(Debug, Clone)]
pub struct Func {
    pub sig: FuncType,
    pub locals: Vec<FuncLocal>,
    pub code: Vec<FuncCode>
}

type Export = (String, usize, ExportType);
type Import = (String, String, ImportType, usize);
type Global = (u8, u8, u32); // (type, mutability, init)
type FuncLocal = (usize, u8);

pub const HEADER_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6D];
pub const HEADER_VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];

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

pub const NONE: u64 = 0x40;
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
pub const I32_GT_U: u8 = 0x4B;
pub const I32_NE: u8 = 0x47;
pub const I32_ADD: u8 = 0x6A;
pub const I32_MUL: u8 = 0x6C;

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
pub const BR: u8 = 0x0C;
pub const BR_IF: u8 = 0x0D;

pub const END: u8 = 0x0B;
pub const RETURN: u8 = 0x0F;
pub const CALL: u8 = 0x10;

pub struct WasmCodeGen {
    funcs: Vec<(usize, Func)>,
    types: Vec<FuncType>,
    exports: Vec<Export>,
    memories: Vec<(u32, u32)>,
    data: Vec<(u32, Vec<u8>)>, // (offset, bytes)
    imports: Vec<Import>,
    globals: Vec<Global>,
}

fn write_name(bytes: &mut Vec<u8>, name: String) {
    write_unsigned_leb128(bytes, name.len() as u64);
    // TODO(sven): is this UTF8?
    bytes.extend(name.into_bytes());
}

fn write_unsigned_leb128(bytes: &mut Vec<u8>, n: u64) {
    leb128::write::unsigned(bytes, n).expect("could not write LEB128");
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
        for imm in &code.immediates {
            write_unsigned_leb128(bytes, *imm);
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
        write_unsigned_leb128(bytes, data.0 as u64); // offset
        bytes.push(END);

        write_vec_len(bytes, &data.1); // vec length
        for b in &data.1 {
            bytes.push(*b);
        }
    }
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

        let expr = vec![
            FuncCode::new1(I32_CONST, *init as u64)
        ];
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
            exports: vec![],
            memories: vec![],
            data: vec![],
            imports: vec![],
            globals: vec![],
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = vec![];

        bytes.extend(&HEADER_MAGIC);
        bytes.extend(&HEADER_VERSION);

        write_section!(bytes, self.types, TYPE_SECTION,
                       write_type_section);
        write_section!(bytes, self.imports, IMPORT_SECTION,
                       write_imports_section);
        write_section!(bytes, self.funcs, FUNCTION_SECTION,
                       write_func_section);
        write_section!(bytes, self.memories, MEMORY_SECTION,
                       write_memory_section);
        write_section!(bytes, self.globals, GLOBAL_SECTION,
                       write_global_section);
        write_section!(bytes, self.exports, EXPORT_SECTION,
                       write_export_section);
        write_section!(bytes, self.funcs, CODE_SECTION, write_code_section);
        write_section!(bytes, self.data, DATA_SECTION, write_data_section);

        bytes
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

    pub fn add_memory(&mut self, min: u32, max: u32) -> usize {
        assert!(self.memories.len() == 0);
        self.memories.push((min, max));

        0
    }

    pub fn add_data(&mut self, offset: u32, bytes: Vec<u8>) -> u32 {
        self.data.push((offset, bytes.clone()));
        bytes.len() as u32
    }

    pub fn add_import(&mut self, module: String, name: String,
                      import_type: ImportType, typeidx: usize) -> usize {
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
