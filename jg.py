import numpy as np
import time
import pandas as pd
from pysat.solvers import Solver

def jg(items: np.ndarray, requirements: np.ndarray):
    oracle_count = 0
    begin_time = time.perf_counter()

    cols = items.shape[1]
    int_size = 32
    ccols = (cols - 1) // int_size + 1

    feasibleBaskets = []
    minimalFeasibleBaskets = []
    rs_fb = []
    rs_mfb = []
    FB_holder = []
    MFB_holder = []
    rows_fb = 0
    rows_mfb = 0
    found_dual = False

    while not found_dual:
        startt = time.perf_counter()

        found_dual, clause = dcheck(FB_holder, MFB_holder)

        endt = time.perf_counter()

        if found_dual:
            print()
            print(f'Final FB size: {rows_fb}. Final mFB size: {rows_mfb}. Time to verify duality: {endt - startt:.4f} seconds.')
        else:
            clong = rfb(clause, cols)  # Uncompress
            oracle_count += 1
            if not is_basket(items, requirements, np.logical_not(clong)):
                # This is a minimal feasible basket (mFB)
                for i in range(cols):
                    if clong[i]:
                        ctest = clong.copy()
                        ctest[i] = False
                        oracle_count += 1
                        if not is_basket(items, requirements, np.logical_not(ctest)):
                            clong = ctest
                clause = rows_to_bits(clong)
                rsize = np.sum(clong)
                ispot = next((i for i, r in enumerate(rs_mfb) if r >= rsize), None)
                if ispot is None:
                    minimalFeasibleBaskets.append(clause)
                    MFB_holder.append(transform_sat(rfb(clause, cols), 0))
                    rs_mfb.append(rsize)
                else:
                    minimalFeasibleBaskets.insert(ispot, clause)
                    MFB_holder.append(ispot, transform_sat(rfb(clause, cols), 0))
                    rs_mfb.insert(ispot, rsize)
                rows_mfb += 1
            else:
                # This is a feasible basket (FB)
                clong = np.logical_not(clong)
                for i in range(cols):
                    if clong[i]:
                        ctest = clong.copy()
                        ctest[i] = False
                        oracle_count += 1
                        if is_basket(items, requirements, ctest):
                            clong = ctest
                clause = rows_to_bits(clong)
                rsize = np.sum(clong)
                ispot = next((i for i, r in enumerate(rs_fb) if r >= rsize), None)
                if ispot is None:
                    feasibleBaskets.append(clause)
                    FB_holder.append(transform_sat(rfb(clause, cols), 1))
                    rs_fb.append(rsize)
                else:
                    feasibleBaskets.insert(ispot, clause)
                    FB_holder.insert(ispot, transform_sat(rfb(clause, cols), 1))
                    rs_fb.insert(ispot, rsize)
                rows_fb += 1

            if (rows_fb + rows_mfb) % 100 == 0:
                print(f'FB size: {rows_fb}. mFB size: {rows_mfb}. Time for latest clause: {endt - startt:.4f} seconds.')

    end_time = time.perf_counter()
    print()
    print(f'Total time spent in generating forms: {end_time - begin_time:.4f} seconds.')
    print(f'Total number of oracle calls: {oracle_count}.')

    return feasibleBaskets, minimalFeasibleBaskets


def rows_to_bits(bitrow):
    """
    Compresses a 1D binary vector into a list of integers,
    where each bit in the output represents a bit in the input.
    
    Args:
        bitrow: 1D array of 0s and 1s.
    
    Returns:
        1D array of integers of encoded bits. 
    """
    int_size = 32
    bitrow = np.asarray(bitrow, dtype=np.uint8).flatten()
    num_bits = len(bitrow)
    num_ints = (num_bits + int_size - 1) // int_size  

    result = np.zeros(num_ints, dtype=np.uint32)

    for j in range(num_ints):
        chunk_size = int_size if j < num_ints - 1 else num_bits - j * int_size
        for k in range(chunk_size):
            idx = j * int_size + k
            if bitrow[idx]:
                result[j] |= (1 << k)  

    return result

import numpy as np

def rfb(n, width=32):
    """
    Convert an integer to a list of 0/1 bits.
    
    Args:
        n: The number to convert.
        width: How many bits to extract.
        
    Returns:
        List: A list of 0s and 1s representing the binary form of inputted number. 
    """
    return [(n >> i) & 1 for i in range(width)]

import numpy as np 

def is_basket(basket_items: np.ndarray, requirements: np.ndarray, boolean_vector: np.ndarray) -> bool:
    """
    Determines if a food basket meets daily nutritional requirements.
    
    Args:
        basket_items: 2D array of shape (nutrients, items) containing nutrient values
        requirements: 1D array of shape (nutrients,) containing daily required values
        boolean_vector: 1D boolean array of shape (items,) indicating selected items
    
    Returns:
        bool: True if basket meets all requirements, False otherwise
    """
    # Multiply items by selection vector and sum nutrients
    basket = basket_items * boolean_vector
    nut_values = np.sum(basket, axis=1)
    
    # Check if all requirements are met
    feasible = np.all(nut_values >= requirements)
    
    return feasible

def dcheck(f: np.ndarray, g: np.ndarray):

    f_and_g = f + g
    solver = Solver(name='Glucose4')
    for clause in f_and_g:
        solver.add_clause(clause)

    result = solver.solve()
    
    if result:
        is_dual = 0
    else:
        is_dual = 1
    
    clause = solver.get_model()
    if clause:
        clause = [1 if x > 0 else 0 for x in clause]
    else:
        clause = 0
        
    return is_dual, clause

def transform_sat(clause, option):
    # This takes in a Boolean list of 0's and 1's (clause)
    # The other input represents whether it is part of the FB list or mFB list
    # If it is part of the mFB list the clause must be multipled by -1 

    size = len(clause)
    vector = np.arange(1, size + 1)

    if option:
        transformed_clause = np.array(clause) * vector
    else:
        transformed_clause = -np.array(clause) * vector

    transformed_list = list(map(int, transformed_clause[transformed_clause != 0]))

    return transformed_list
