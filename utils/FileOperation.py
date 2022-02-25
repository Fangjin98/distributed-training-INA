def write_tensor_to_file(filename, tensor):
    with open(filename, 'w') as f:
        for t in tensor:
            f.write(str(float(t)) + ' ')
    print("Write done.")
