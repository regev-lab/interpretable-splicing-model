import argparse
import subprocess
from typing import Generator, Tuple, List, Iterable, Optional, Dict


def RNAsubopt(
    seq: str, RNAsubopt_bin: str = "RNAsubopt", delta_energy: float = 5.0
) -> List[Tuple[str, float]]:
    """
    Uses RNAsubpot to return a list of secondary structures
    and their associated energies in kcal/mol.
    """
    try:
        cmd = " ".join(
            [
                "bash -c '",
                f"{RNAsubopt_bin}",
                f"--deltaEnergy {delta_energy}",
                f"-i <(echo {seq})" "'",
            ]
        )
        out = subprocess.check_output(cmd, shell=True).decode()

        typify = lambda t: (str(t[0]), float(t[1]))
        processed = [typify(line.split()) for line in out.split("\n")[1:-1]]
        return processed

    except subprocess.CalledProcessError as exc:
        print(exc)
        return []

def RNAsample(
    seqs: List[str], RNAfold_bin: str = "RNAsubopt", temperature: float = 37.0, num_structs: int = 5, maxBPspan: int = 0
) -> List[List[str]]:
    """
    Uses RNAsubopt to sample num_structs structures from the Boltzmann distribution. Accepts a list of sequences, and calls RNAsubopt in a batched way on them.
    """
    try:
        input_binary = b"\n".join([seqs[i].encode('ascii') for i in range(len(seqs))]) # convert to binary

        out = subprocess.check_output([RNAfold_bin, f"--stochBT={num_structs}", f"-T {temperature}"] +
                                      ([f"--maxBPspan={maxBPspan}"]  if maxBPspan != 0 else []), 
                                      input=input_binary).decode().split("\n")[:-1] # remove the new line at the end

        assert(len(out) == (num_structs+1)*len(seqs))
        
        ret_value = [  out[(num_structs+1)*i+1:(num_structs+1)*(i+1)] for i in range(len(seqs))]

        return ret_value

    except subprocess.CalledProcessError as exc:
        print(exc)
        return 0.0



def RNAfold(
    seqs: List[str], RNAfold_bin: str = "RNAfold", temperature: float = 37.0, maxBPspan: int = 0, commands_file: str = ""
) -> (str, float):
    """
    Uses RNAfold to return MFE energy. Accepts a list of sequences, and calls RNAfold in a batched way on them.
    """
    try:
        input_binary = b"\n".join([seqs[i].encode('ascii') for i in range(len(seqs))]) # convert to binary

        out = subprocess.check_output([RNAfold_bin, "-j 8", "--noPS", f"-T {temperature}"] + 
            (["--commands="+commands_file]  if commands_file != "" else []) +
            ([f"--maxBPspan={maxBPspan}"]  if maxBPspan != 0 else []), 
            input=input_binary).decode().split("\n")

        mfe_lines = [out[i].split(maxsplit=1) for i in range(1,len(out),2)]
        
        ret_value = [[mfe_lines[i][0], float(mfe_lines[i][1].strip('() '))] for i in range(len(mfe_lines))]

        return ret_value

    except subprocess.CalledProcessError as exc:
        print(exc)
        return 0.0


def RNA_partition_function(
    seqs: List[str], constraints: List[str], RNAfold_bin: str = "RNAfold", temperature: float = 37.0, commands_file: str = ""
) -> (str, float):
    """
    Uses RNAfold to compute partition function with constraints on the structure
    """
    assert(len(seqs)==len(constraints))
    try:
        input_binary = b"\n".join([(seqs[i]+"\n"+constraints[i]).encode('ascii') for i in range(len(seqs))]) # convert to binary

        out = subprocess.check_output([RNAfold_bin, "-j 8",  "--noPS", "--noDP", "-p0" ,"-C", f"-T {temperature}"] + (["--commands="+commands_file]  if commands_file != "" else []), input=input_binary).decode().split("\n")

        ret_value = [float(out[i].split(sep=" ")[-2]) for i in range(2,len(out),4)] # take the third line in each group of 4 lines
        
        return ret_value

    except subprocess.CalledProcessError as exc:
        print(exc)
        return 0.0




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=str,
        default="AGUAGUCUAGUAUGACUGUA",
        help="RNA sequence to process",
    )
    args = parser.parse_args()
    output = RNAsubopt(seq=args.seq)
    print(output)
