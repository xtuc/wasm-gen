pub struct Naming {
    pub index: u32,
    pub name: String,
}

pub fn write_var_uint32(v: u32, bytes: &mut Vec<u8>) {
    let mut buf = [0u8; 1];
    let mut v = v;
    loop {
        buf[0] = (v & 0b0111_1111) as u8;
        v >>= 7;
        if v > 0 {
            buf[0] |= 0b1000_0000;
        }
        bytes.extend_from_slice(&mut buf[..]);
        if v == 0 {
            break;
        }
    }
}

pub fn write_var_uint7(v: u8, bytes: &mut Vec<u8>) {
    // TODO: check range
    bytes.push(v);
}
