""" 
Running this file without any AssertionErrors verifies every claim in our paper that
references this github repository. 
"""

from saturation import Matrix

Q2 = Matrix.from_str_list([
    "1010",
    "1000",
    "0001",
    "0100"
])

Q4 = Matrix.from_str_list([
    "1010",
    "1000",
    "0001",
    "1100"
])

Q6 = Matrix.from_str_list([
    "1010",
    "1000",
    "0001",
    "1101"
])

Q7 = Matrix.from_str_list([
    "1010",
    "1000",
    "0001",
    "0101"
])

P2 = Matrix.from_str_list([
    "1011",
    "1101"
])

W_V_Q2 = Matrix(Matrix.from_str_list([
    "101000",
    "111010",
    "000000",
    "010001",
    "000100"
]))

W_H_Q2 = Matrix(Matrix.from_str_list([
    "101010",
    "011000",
    "110001",
    "010000",
    "000010",
    "001000"
]))

W_Q4 = Matrix(Matrix.from_str_list([
    "0000001001010",
    "0000000101000",
    "0000001100001",
    "1010000000000",
    "1110100000000",
    "0000000000000",
    "1100010000000",
    "0111000000000",
    "1010000000000",
    "0000000111000",
    "0000000101010",
    "0000000101000",
    "0000001001000"
]))

W_V_Q6 = Matrix(Matrix.from_str_list([
    "1010000",
    "1110100",
    "0000000",
    "1101111",
    "0111010",
    "1010001"
]))

W_H_Q6 = Matrix(Matrix.from_str_list([
    "1010100",
    "0110000",
    "0100010",
    "0110010",
    "1010111",
    "0110110",
    "1010101"
]))

W_V_Q7 = Matrix(Matrix.from_str_list([
    "101000",
    "111010",
    "000000",
    "010111",
    "000101"
]))

W_H_Q7 = Matrix(Matrix.from_str_list([
    "1010100",
    "0110000",
    "0100011",
    "1100010",
    "0000110",
    "0010101"
]))

W2 = Matrix(Matrix.from_str_list([
    "011011",
    "111010",
    "001011",
    "101011"
]))

if __name__ == "__main__":
    assert W_V_Q2.is_vertical_witness(Q2, 2)
    assert W_H_Q2.is_horizontal_witness(Q2, 3)
    assert W_Q4.is_witness(Q4, 5, 10)
    assert W_V_Q6.is_vertical_witness(Q6, 2)
    assert W_H_Q6.is_horizontal_witness(Q6, 3)
    assert W_V_Q7.is_vertical_witness(Q7, 2)
    assert W_H_Q7.is_horizontal_witness(Q7, 3)
    assert not W2.contains_pattern(P2)
