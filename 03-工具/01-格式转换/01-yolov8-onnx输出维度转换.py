import onnx
import onnx.helper as helper
import sys
import os

def main():

    if len(sys.argv) < 2:
        print("Usage:\n python v8trans.py yolov8n.onnx")
        return 1

    file = sys.argv[1] # onnx模型地址
    if not os.path.exists(file):
        print(f"Not exist path: {file}")
        return 1

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix # 转换后的模型名称

    model = onnx.load(file) # 加载模型

    # 修改模型的输出
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    # 下面是核心代码，不同版本的yolov8-seg模型需要做调整。
    # 这个是yolov8n的修改方式，其他版本的模型需要自己实现输出修改逻辑
    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return 0

if __name__ == "__main__":
    sys.exit(main())