def to_bitmask(n, bits):
        # [2:] to chop off the "0b" part
        mask = [int(digit) for digit in bin(n)[2:]] 
        # pad to fixed length
        return [0] * (bits - len(mask)) + mask 
    
def invert_mask(mask):
    return [int(not bit) for bit in mask]