#sudoku_solver.py
def es_valido(sudoku, fila, col, num):
    # Verifica si el número ya está en la fila
    if num in sudoku[fila]:
        return False
    # Verifica si el número ya está en la columna
    for i in range(9):
        if sudoku[i][col] == num:
            return False
    # Verifica si el número ya está en el subcuadrante 3x3
    start_row, start_col = 3 * (fila // 3), 3 * (col // 3)
    for i in range(3):
        for j in range(3):
            if sudoku[start_row + i][start_col + j] == num:
                return False
    return True

def resolver_sudoku(sudoku):
    # Encuentra una celda vacía (representada por 0)
    for fila in range(9):
        for col in range(9):
            if sudoku[fila][col] == 0:
                for num in range(1, 10):
                    # Si es válido, intenta colocar el número
                    if es_valido(sudoku, fila, col, num):
                        sudoku[fila][col] = num
                        # Intenta resolver el resto del Sudoku
                        if resolver_sudoku(sudoku):
                            return True
                        # Si no funciona, vacía la casilla y retrocede
                        sudoku[fila][col] = 0
                return False
    return True

def imprimir_sudoku(sudoku):
    for fila in sudoku:
        print(" ".join(str(num) if num != 0 else '.' for num in fila))

## Ejemplo de uso
#sudoku = [
#    [5, 3, 0, 0, 7, 0, 0, 0, 0],
#    [6, 0, 0, 1, 9, 5, 0, 0, 0],
#    [0, 9, 8, 0, 0, 0, 0, 6, 0],
#    [8, 0, 0, 0, 6, 0, 0, 0, 3],
#    [4, 0, 0, 8, 0, 3, 0, 0, 1],
#    [7, 0, 0, 0, 2, 0, 0, 0, 6],
#    [0, 6, 0, 0, 0, 0, 2, 8, 0],
#    [0, 0, 0, 4, 1, 9, 0, 0, 5],
#    [0, 0, 0, 0, 8, 0, 0, 7, 9]
#]
#
#if resolver_sudoku(sudoku):
#    print("Sudoku solucionado:")
#    imprimir_sudoku(sudoku)
#else:
#    print("No tiene solución.")
#