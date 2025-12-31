import graphzero as gz
import time

t0 = time.time()
gz.convert_csv_to_gl("papers100M_edges.csv", "papers100M.gl", directed=True)
t1 = time.time()
print(f"total time {t1-t0:.2f} s")