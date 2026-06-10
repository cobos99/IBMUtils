from qiskit import qpy
import base64
import io

def load_circuit_from_job(job):
    
    items = job.inputs["quantum_program"].items
    circuits = items[0].circuit

    #circuits = []
    #for item in items:
    #    cs = item["circuit"]["circuit_b64"]
    #    raw = base64.b64decode(cs)
    #    circuits.append(qpy.load(io.BytesIO(raw))[0])
    
    
    return circuits