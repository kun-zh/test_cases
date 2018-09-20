import tvm

def intrin_gemv(m, n):
    w = tvm.placeholder((m, n), name='w')
    x = tvm.placeholder((n,), name='x')
    k = tvm.reduce_axis((0, n), name='k')
    z = tvm.compute((m,), lambda i:
                    tvm.sum(w[i, k] * x[k], axis=k), name='z')
    Wb = tvm.decl_buffer(w.shape, w.dtype,
                         name="W",
						 scope="local",
                         offset_factor=16,
                         strides=[tvm.var('ldw'), 1])
    def intrin_func(ins, outs):
        ww, xx = ins
        zz = outs[0]
        ww_ptr = ww.access_ptr("r")
        xx_ptr = xx.access_ptr("r")
        zz_ptr = zz.access_ptr("w")
        body = tvm.call_extern(
            "int32","gemv", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        reset = tvm.call_extern(
            "int32","fill_zero", zz_ptr, n)
        update = tvm.call_extern(
            "int32","gemv_add", ww_ptr, xx_ptr, zz_ptr, n, ww.strides[0])
        return body, reset, update

    with tvm.build_config(data_alignment=16,
                          offset_factor=16):
        return tvm.decl_tensor_intrin(z.op, intrin_func,
                                      binds={w: Wb})

def test_tensorize_matmul():
    n = 1024
    m = n
    l = n
    A = tvm.placeholder((n, l), name='A')
    B = tvm.placeholder((m, l), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute((n, m), lambda i, j:
                    tvm.sum(B[j, k] * A[i, k], axis=k), name='C')
    s = tvm.create_schedule(C.op)
    BL = s.cache_read(B, "local", [C])
    x, y = C.op.axis
    yo, yi = s[C].split(y, 16)
    gemv = intrin_gemv(16, l)
    s[C].tensorize(yi, gemv)
    s = s.normalize()
    tvm.lower(s, [A, B, C])
    print(tvm.lower(s, [A, B, C], simple_mode=True))



from te import tvm
import te.lang.cce
import topi 
from topi.cce import util

def caffe_relu_layer_cce(shape, dtype, negative_slope= 0, 
			 kernel_name = "caffe_relu_layer_cce"):
    util.check_shape_rule(shape)
    inp_dtype = dtype.lower()    

    with te.target.cce():
        data = tvm.placeholder(shape, name="data", dtype=inp_dtype)
        slope_tmp = tvm.const(negative_slope, dtype = inp_dtype)
        tmp = te.lang.cce.vmuls(data, slope_tmp)
        res_tmp = te.lang.cce.vmax(tmp, data)    
        res = te.lang.cce.cast_to(res_tmp, inp_dtype)
        sch = generic.auto_schedule(res)

    config = {"print_ir" : need_print, 
	      "need_build" : need_build, 
	      "name": kernel_name, 
	      "cce_path" : cce_path, 
	      "tensor_list" : [data, res]}

    te.lang.cce.cce_build_code(sch, config)

	
if __name__ == "__main__":
    test_tensorize_matmul()
