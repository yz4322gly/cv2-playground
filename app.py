import cv2
import gradio as gr

from web_fn import cv2_cvt_color, cv2_split, cv2_erode, cv2_dilate, cv2_roi, cv2_resize, cv2_threshold, cv2_open, cv2_close, cv2_gradient, \
    cv2_tophat, cv2_blackhat, cv2_blur, cv2_box_filter, cv2_gaussian_blur, cv2_median_blur, cv2_pyr_down, cv2_sobel, cv2_scharr, cv2_laplacian, cv2_canny, \
    cv2_contours

with gr.Blocks() as cv_playground:
    uploaded_image = None
    rows = []
    history_imgs = []
    history_texts = []


    def get_show_str(his_list):
        """
        从历史记录列表到显示的转换方式
        Args:
            his_list:

        Returns:

        """
        return "\n".join(history_texts)


    def process_image(image_file):
        global uploaded_image
        global history_texts
        # 加载图像
        uploaded_image = cv2.imread(image_file)
        history_texts = ["初始图像"]
        return uploaded_image, "初始图像"


    def reset_image():
        """重置到原始图像"""
        global uploaded_image
        global history_texts
        history_texts = ["初始图像"]
        return uploaded_image, "初始图像"


    def undo_image():
        """
        使用history_imgs向之前结果撤销
        Returns:
        """
        global history_imgs
        global uploaded_image
        global history_texts
        his_len = len(history_imgs)
        if his_len > 1:
            history_texts.pop()
            return history_imgs.pop(), get_show_str(history_texts)
        else:
            gr.Warning("撤销到原始图像啦!")
            return uploaded_image, get_show_str(history_texts)


    def handle_wrapper(func):
        """
        包装器方法,记录图像历史,处理完成记录到list上
        Args:
            func:

        Returns:

        """
        global history_imgs
        global history_texts

        def wrapper(*args, **kwargs):
            history_imgs.append(args[0])
            if len(args) > 1:
                history_texts.append(func.__name__ + ":" + str(args[1:]))
            else:
                history_texts.append(func.__name__)
            fn_rs = func(*args, **kwargs)
            if isinstance(fn_rs, tuple):
                return fn_rs + (get_show_str(history_texts),)
            else:
                return func(*args, **kwargs), get_show_str(history_texts)

        return wrapper


    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown(value="历史记录")
            his_textbox = gr.Textbox(scale=1, interactive=False, label="操作历史记录", container=False, max_lines=65535)
        with gr.Column(scale=7):
            image_input = gr.File(label="Upload Image")
            show_image = gr.Image()
            image_input.upload(process_image, image_input, [show_image, his_textbox])
        with gr.Column(scale=3):
            with gr.Row():
                reset_button = gr.Button(scale=1, value="重置图像")
                reset_button.click(fn=reset_image, outputs=[show_image, his_textbox])
                undo_button = gr.Button(scale=1, value="撤销(Undo)")
                undo_button.click(fn=undo_image, outputs=[show_image, his_textbox])
            with gr.Row():
                handel_down = gr.Dropdown(scale=1, label="处理方式", value=0, choices=[
                    ("0:色彩通道:转换", 0),
                    ("1:色彩通道:切分", 1),
                    ("2:尺寸:resize", 2),
                    ("3:尺寸:自由裁剪", 3),
                    ("4:尺寸:边缘填充(x)", 4),
                    ("5:图像权重混合(x)", 5),
                    ("6:腐蚀", 6),
                    ("7:膨胀", 7),
                    ("8:阈值处理", 8),
                    ("9:形态学:开运算", 9),
                    ("10:形态学:闭运算", 10),
                    ("11:形态学:梯度运算", 11),
                    ("12:形态学:礼帽", 12),
                    ("13:形态学:黑帽", 13),
                    ("14:滤波:均值滤波", 14),
                    ("15:滤波:方框滤波", 15),
                    ("16:滤波:高斯滤波", 16),
                    ("17:滤波:中值滤波", 17),
                    ("18:采样:高斯金字塔下采样", 18),
                    ("19:边缘检测:Sobel", 19),
                    ("20:边缘检测:Scharr", 20),
                    ("21:边缘检测:Laplacian", 21),
                    ("22:边缘检测:Canny", 22),
                    ("23:轮廓检测:Contours", 23),
                    ("24:模板匹配(x)", 24),
                ])


            @gr.render(inputs=handel_down)
            def handel_down_change(handel_code):
                if handel_code == 0:
                    gr.Markdown("""
                                ## 转换色彩通道
                                图像的读取使用的是opencv的读取方式,故而是BGR顺序
                                但是图像控件默认以RBG顺序显示,所以可能色彩会偏蓝
                                可以直接转换一次色彩通道调整
                                """)
                    with gr.Row():
                        cvt_code = gr.Dropdown(scale=4, label="转换色彩通道", value="cv.COLOR_BGR2RGB", choices=[
                            "cv.COLOR_BGR2RGB",
                            "cv.COLOR_RGB2BGR",
                            "cv.COLOR_BGR2GRAY",
                            "cv.COLOR_RGB2GRAY",
                        ])
                        cvt_color_btn = gr.Button(scale=1, value="转换色彩通道")
                        cvt_color_btn.click(fn=handle_wrapper(cv2_cvt_color), inputs=[show_image, cvt_code], outputs=[show_image, his_textbox])
                if handel_code == 1:
                    with gr.Row():
                        split_code = gr.Dropdown(scale=4, label="切分色彩通道", value=0, choices=[
                            ("Red(第一通道)", 0),
                            ("Green(第二通道)", 1),
                            ("Blue(第三通道)", 2)])
                        split_btn = gr.Button(scale=1, value="切分色彩通道")
                        split_btn.click(fn=handle_wrapper(cv2_split), inputs=[show_image, split_code], outputs=[show_image, his_textbox])
                if handel_code == 2:
                    gr.Markdown("resize使用指定尺寸或指定比例,使用比例时,尺寸请记为(0,0),否则按指定长宽缩放(可以超过原始尺寸)")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                width = gr.Number(step=1, minimum=0, label="width", info="指定宽度缩放", value=0)
                                height = gr.Number(step=1, minimum=0, label="height", info="指定高度缩放", value=0)
                            with gr.Row():
                                fx = gr.Number(step=0.01, minimum=0, label="fx", info="x缩放比例", value=0.5)
                                fy = gr.Number(step=0.01, minimum=0, label="fy", info="y缩放比例", value=0.5)
                            resize_interpolation = gr.Dropdown(scale=4, label="插值法", value="cv.INTER_LINEAR", choices=[
                                ("最近邻插值", "cv.INTER_NEAREST"),
                                ("双线性插值(默认)", "cv.INTER_LINEAR"),
                                ("像素区域重采样(适用缩小)", "cv.INTER_AREA"),
                                ("双三次插值(适用放大)", "cv.INTER_CUBIC"),
                                ("兰索斯插值(Lanczos)", "cv.INTER_LANCZOS4"),
                                ("位精确双线性插值", "cv.INTER_LINEAR_EXACT"),
                                ("位精确最近邻插值", "cv.INTER_NEAREST_EXACT"),
                            ])
                        with gr.Column():
                            resize_btn = gr.Button(scale=1, value="Resize")
                            resize_btn.click(fn=handle_wrapper(cv2_resize), inputs=[show_image, width, height, fx, fy, resize_interpolation],
                                             outputs=[show_image, his_textbox])
                if handel_code == 3:
                    gr.Markdown("裁剪注意尺寸:width>=x2>x1,height>=y2>y1")
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                x1 = gr.Number(step=1, minimum=0, label="x1", value=0)
                                y1 = gr.Number(step=1, minimum=0, label="y1", value=0)
                            with gr.Row():
                                x2 = gr.Number(step=1, minimum=0, label="x2", value=1920)
                                y2 = gr.Number(step=1, minimum=0, label="y2", value=1080)
                        with gr.Column():
                            roi_btn = gr.Button(scale=1, value="自由裁剪")
                            roi_btn.click(fn=handle_wrapper(cv2_roi), inputs=[show_image, x1, y1, x2, y2], outputs=[show_image, his_textbox])
                if handel_code == 4:
                    gr.Markdown("施工中(未完成)......")
                if handel_code == 5:
                    gr.Markdown("施工中(未完成)......")
                if handel_code == 6:
                    gr.Markdown("""
                          ## 腐蚀(erode)
                          腐蚀操作的主要目的是去除图像中的小对象，并在边界处减少像素点。这通常用于去除噪声、分离连接在一起的物体，或者在图像预处理阶段为其他操作做准备。  
                          腐蚀操作的基本思想是将图像中的每个像素点替换为该点邻域内的最小值。具体来说，对于图像中的每个像素点，`erode`函数会考虑该点周围的一个特定形状和大小的邻域(称为结构元素或核)，然后将该点的值替换为该邻域内的最小值。这个过程会遍历图像中的每个像素点，最终得到一个腐蚀后的图像。  
                          **注意,腐蚀操作的输入一般为灰度图,多通道图像针对每一通道处理**
                          """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        erode_btn = gr.Button(scale=1, value="腐蚀")
                        erode_btn.click(fn=handle_wrapper(cv2_erode), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 7:
                    gr.Markdown("""
                          ## 膨胀(dilate)
                          主要用于扩大图像中对象的边界。这种操作在处理二值图像时特别有用，比如在去除小的白色噪声、连接邻近的对象或者使对象的边界更加突出时。  
                          膨胀操作是通过使用一个结构元素(或称为核)来完成的，这个结构元素在图像上滑动，并将结构元素覆盖下的像素替换为结构元素覆盖下的最大值(对于二值图像，通常是白色像素)。这个过程导致图像中的对象边界向外扩展。  
                          **注意,膨胀操作的输入一般为灰度图,多通道图像针对每一通道处理**
                          """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        dilate_btn = gr.Button(scale=1, value="膨胀")
                        dilate_btn.click(fn=handle_wrapper(cv2_dilate), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 8:
                    gr.Markdown("""## 阈值操作 """)
                    with gr.Column():
                        with gr.Row():
                            thresh = gr.Slider(0, 255, step=0.1, value=50, label="thresh", info="阈值")
                            maxval = gr.Slider(0, 255, step=0.1, value=255, label="maxval", info="最大值(详见阈值方法)")
                        threshold_type = gr.Dropdown(label="阈值方法", value="cv.THRESH_BINARY", choices=[
                            ("THRESH_BINARY:二值法,大于阈值的设置为maxval,否则为0", "cv.THRESH_BINARY"),
                            ("THRESH_BINARY_INV:反二值法,小于阈值的设置为maxval,否则为0", "cv.THRESH_BINARY_INV"),
                            ("THRESH_TRUNC:截断法,大于阈值的设置为阈值,否则不变", "cv.THRESH_TRUNC"),
                            ("THRESH_TOZERO:低阈值置零法,大于阈值的保持不变,否则设置为0", "cv.THRESH_TOZERO"),
                            ("THRESH_TOZERO_INV:高阈值置零法,小于阈值的保持不变,否则设置为0", "cv.THRESH_TOZERO_INV")
                        ])
                        threshold_btn = gr.Button(value="阈值操作")
                        threshold_btn.click(fn=handle_wrapper(cv2_threshold), inputs=[show_image, thresh, maxval, threshold_type],
                                            outputs=[show_image, his_textbox])
                if handel_code == 9:
                    gr.Markdown("""
                            ## 开运算(Opening)
                            开运算是形态学变换中的一种操作，主要用于去除小对象(通常被视为噪声)而不显著改变较大对象的面积。这种操作在处理二值图像时特别有效，比如在去除小的黑色斑点、断开细小连接或者平滑较大对象的边界时。   
                            开运算首先通过**腐蚀**操作来缩小图像中的对象，紧接着使用与**腐蚀**操作相同的结构元素进行膨胀操作。这种先腐蚀后膨胀的组合操作能够有效地去除那些无法通过单独腐蚀或膨胀完全去除的小对象，同时保持较大对象的形状和位置相对不变。  
                            **注意，开运算的输入一般为二值图像，但也可以应用于灰度图像，其中针对灰度图像，开运算会针对每个像素值进行腐蚀和膨胀操作，不过这种用法相对较少。**
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        open_btn = gr.Button(scale=1, value="开运算")
                        open_btn.click(fn=handle_wrapper(cv2_open), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 10:
                    gr.Markdown("""
                            ## 闭运算(Closing)
                            闭运算是形态学变换中的另一种操作，与开运算相反，它主要用于填充图像内的小型黑洞(即前景区域中的黑色区域)和连接邻近的对象。这种操作在处理二值图像时尤其有用，比如在填补对象内部的孔洞、连接邻近的断裂部分或平滑对象的边界时。  
                            闭运算首先通过**膨胀**操作来扩大图像中的对象，紧接着使用与**膨胀**操作相同的结构元素进行腐蚀操作。这种先膨胀后腐蚀的组合操作可以填充对象内部的小孔，并平滑对象的边界，同时保持对象的基本形状和位置不变。  
                            **同样，闭运算的输入一般为二值图像，但也可以应用于灰度图像，其中针对灰度图像，闭运算会针对每个像素值进行膨胀和腐蚀操作。**
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        close_btn = gr.Button(scale=1, value="闭运算")
                        close_btn.click(fn=handle_wrapper(cv2_close), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 11:
                    gr.Markdown("""
                            ## 形态学梯度(Gradient)
                            形态学梯度是通过计算膨胀后的图像与腐蚀后的图像之间的差异来得到的。具体来说，形态学梯度定义为膨胀后的图像与腐蚀后的图像之差  
                            形态学梯度在图像处理中常用于边缘检测，尤其是在处理二值图像时。它能够突出显示图像中形状变化明显的区域，即边缘部分。由于形态学梯度对噪声的敏感度较低，因此在边缘检测中具有一定的优势。
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        gradient_btn = gr.Button(scale=1, value="形态学梯度")
                        gradient_btn.click(fn=handle_wrapper(cv2_gradient), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 12:
                    gr.Markdown("""
                            ## 礼帽(Top Hat)
                            礼帽操作也被称为“白帽”(White Hat)，它用于突出图像中比周围区域更亮的部分，通常用于检测图像中的小亮点或光斑。 (毛刺输出) 
                            礼帽操作是通过从原始图像中减去其开运算结果来实现的。  
                            开运算是先腐蚀后膨胀的过程，它有助于去除小的亮对象并平滑较大对象的边界，而不明显改变其面积。
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        tophat_btn = gr.Button(scale=1, value="礼帽")
                        tophat_btn.click(fn=handle_wrapper(cv2_tophat), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])
                if handel_code == 13:
                    gr.Markdown("""
                            ## 黑帽(Black Hat)
                            黑帽操作用于突出图像中比周围区域更暗的部分，通常用于检测图像中的小黑点或暗斑。  
                            黑帽操作是通过从原始图像的闭运算结果中减去原始图像来实现的。  
                            闭运算是先膨胀后腐蚀的过程，它有助于填充图像中的小黑洞，连接邻近对象，并平滑对象的轮廓。
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="核大小(可以为矩形,但一般是正方形,此处指示边长)",
                                                value=2)
                        iterations = gr.Number(scale=2, minimum=1, label="iterations", info="迭代次数", value=1)
                        blackhat_btn = gr.Button(scale=1, value="黑帽")
                        blackhat_btn.click(fn=handle_wrapper(cv2_blackhat), inputs=[show_image, kernel_size, iterations], outputs=[show_image, his_textbox])

                if handel_code == 14:
                    gr.Markdown("""
                            ## 均值滤波(blur)
                            均值滤波是一种简单的图像平滑技术，用于减少图像中的噪声和细节。  
                            它通过计算图像中每个像素及其邻域内像素的平均值，并将该平均值作为该像素的新值来工作。  
                            均值滤波能够有效地抑制随机噪声，但也可能导致图像的边缘和细节变得模糊。
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=2, minimum=0, label="kernel_size",
                                                info="滤波核的大小，通常是正奇数((width, height))。宽度和高度可以不同，但通常设置为相同的值以创建一个方形核。",
                                                value=2)
                        blur_btn = gr.Button(scale=1, value="均值滤波")
                        blur_btn.click(fn=handle_wrapper(cv2_blur), inputs=[show_image, kernel_size], outputs=[show_image, his_textbox])
                if handel_code == 15:
                    gr.Markdown("""
                            ## 方框滤波(boxFilter)
                            方框滤波是均值滤波的一种特殊情况，其中滤波器的系数全部相等(通常为1)，且滤波器的响应被其系数的总和(即滤波器的大小)除以，以实现归一化。  
                            实质上，方框滤波就是简单的均值滤波，只是有时通过调整归一化系数，可以实现不同的效果，如保持图像的整体亮度不变。  
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(scale=1, minimum=0, label="kernel_size",
                                                info="滤波核的大小，通常是正奇数((width, height))。宽度和高度可以不同，但通常设置为相同的值以创建一个方形核。",
                                                value=2)
                        ddepth = gr.Number(scale=1, minimum=-1, label="ddepth", visible=False,
                                           info=" 输出图像的所需深度(数据类型)。当设置为-1时，表示输出图像将与输入图像具有相同的深度。这个参数在方框滤波中不是特别关键，因为结果通常是归一化的像素平均值。",
                                           value=-1)
                        normalize = gr.Radio(label="normalize", choices=[("归一化", 1), ("不归一化", 0)],
                                             info="指定是否对核内的值进行归一化。如果设置为True(默认值)，则内核内的值将被归一化(即除以核内元素的数量)，从而得到平均值。如果设置为False，则不进行归一化，滤波结果将是核内像素值的和。",
                                             value=1)
                        box_filter_btn = gr.Button(scale=1, value="方框滤波")
                        box_filter_btn.click(fn=handle_wrapper(cv2_box_filter), inputs=[show_image, kernel_size, ddepth, normalize],
                                             outputs=[show_image, his_textbox])
                if handel_code == 16:
                    gr.Markdown("""
                            ## 高斯滤波(GaussianBlur)
                            高斯滤波是一种在图像处理中广泛使用的平滑技术，它基于高斯函数计算图像中每个像素的加权平均值。  
                            与均值滤波不同，高斯滤波在计算平均值时，给每个邻域像素分配了不同的权重，这些权重由高斯函数根据像素与中心像素的距离决定。  
                            因此，高斯滤波在平滑图像的同时，能够更好地保留图像的边缘信息，减少模糊效应。
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(minimum=3, label="kernel_size", step=2,
                                                info="滤波核的大小，必须是正奇数((width, height))。宽度和高度可以不同，但通常设置为相同的值以创建一个方形核。",
                                                value=3)
                        sigmaX = gr.Number(minimum=0, label="sigmaX",
                                           info="X方向上的高斯核标准偏差。如果sigmaY 与 sigmaX 两者都为零，则它们从核大小中计算得出。较大的 sigma 值意味着更模糊的图像。",
                                           value=0)
                        sigmaY = gr.Number(minimum=0, label="sigmaY",
                                           info="Y方向上的高斯核标准偏差。如果sigmaY 与 sigmaX 两者都为零，则它们从核大小中计算得出。较大的 sigma 值意味着更模糊的图像。",
                                           value=0)
                        dst = gr.Number(visible=False, scale=1, minimum=-1, label="dst", info="迭代次数", value=-1)
                        gaussian_blur_btn = gr.Button(value="高斯滤波")
                        gaussian_blur_btn.click(fn=handle_wrapper(cv2_gaussian_blur), inputs=[show_image, kernel_size, sigmaX, dst, sigmaY],
                                                outputs=[show_image, his_textbox])
                if handel_code == 17:
                    gr.Markdown("""
                            ## 中值滤波(medianBlur)
                            中值滤波是一种非线性滤波技术，用于去除图像中的噪声，特别是椒盐噪声(salt-and-pepper noise)。  
                            中值滤波不是通过计算像素的平均值来工作的，而是通过将每个像素的值替换为其邻域内像素值的中位数来实现的。  
                            这种方法对于去除随机噪声非常有效，同时能够很好地保持图像的边缘信息，因为它基于排序而不是平均。因此，中值滤波在处理具有尖锐边缘的图像时，比均值滤波和高斯滤波更为优越。  
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(minimum=3, label="kernel_size", step=2,
                                                info="滤波核的大小，必须是正奇数,中值滤波为方形核。",
                                                value=3)
                        median_blur_btn = gr.Button(scale=1, value="中值滤波")
                        median_blur_btn.click(fn=handle_wrapper(cv2_median_blur), inputs=[show_image, kernel_size], outputs=[show_image, his_textbox])
                if handel_code == 18:
                    gr.Markdown("""
                            ## 高斯金字塔下采样(pyrDown)
                            用于对图像进行高斯金字塔下采样。这个函数通过先对图像进行高斯模糊，然后删除图像的偶数行和列来实现图像尺寸的减半。  
                            这样做的结果是一个尺寸减半但保持图像基本特征的图像，这对于图像金字塔的构建、图像的多尺度表示以及许多图像处理任务(如特征提取、图像匹配等)非常有用。
                            """)
                    with gr.Row():
                        width = gr.Number(visible=False, step=1, minimum=0, label="width",
                                          info="指定宽度缩放,如果指定了 dstsize(输出图像的大小)，则确保输出图像的大小是输入图像大小的一半(或非常接近一半，误差不超过2个像素)。但是，通常在使用 cv2.pyrDown() 时不会直接指定 dstsize，而是让函数自动计算。",
                                          value=0)
                        height = gr.Number(visible=False, step=1, minimum=0, label="height",
                                           info="指定高度缩放,如果指定了 dstsize(输出图像的大小)，则确保输出图像的大小是输入图像大小的一半(或非常接近一半，误差不超过2个像素)。但是，通常在使用 cv2.pyrDown() 时不会直接指定 dstsize，而是让函数自动计算。",
                                           value=0)
                        pyr_down_btn = gr.Button(scale=1, value="高斯金字塔下采样")
                        pyr_down_btn.click(fn=handle_wrapper(cv2_pyr_down), inputs=[show_image, width, height], outputs=[show_image, his_textbox])
                if handel_code == 19:
                    gr.Markdown("""
                            ## 边缘检测:Sobel
                            Sobel边缘检测通过应用两个3x3的卷积核（分别对应水平和垂直方向的梯度计算），对图像的每个像素进行卷积操作，然后计算梯度幅度，并通过设定阈值来判定边缘点，最终生成边缘图像。  
                            只使用了水平和垂直两个方向的模板，对于纹理较为复杂的图像，边缘检测效果可能不理想。  
                             $$
                            G_x = \\begin{bmatrix}
                            -1 & 0 & 1 \\\\
                            -2 & 0 & 2 \\\\
                            -1 & 0 & 1
                            \\end{bmatrix}
                            $$
                            $$
                            G_y = \\begin{bmatrix}
                            1 & 2 & 1 \\\\
                            0 & 0 & 0 \\\\
                            -1 & -2 & -1
                            \\end{bmatrix}
                            $$
                            **这里值提供了基础参数使用**
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(minimum=3, label="kernel_size", step=2,
                                                info="算子核的大小，正奇数。",
                                                value=3)
                        x_alpha = gr.Slider(0, 1, step=0.01, value=0.5, label="x_alpha", info="x混合比例")
                        y_alpha = gr.Slider(0, 1, step=0.01, value=0.5, label="y_alpha", info="y混合比例")
                        sobel_btn = gr.Button(scale=1, value="边缘检测:Sobel")
                        sobel_btn.click(fn=handle_wrapper(cv2_sobel), inputs=[show_image, kernel_size, x_alpha, y_alpha], outputs=[show_image, his_textbox])
                if handel_code == 20:
                    gr.Markdown("""
                            ## 边缘检测:scharr
                            Scharr算子是用于图像边缘检测的一种算子，它通过对图像进行卷积运算来检测边缘。Scharr算子包含两个3x3的卷积核，分别用于计算图像在水平方向和垂直方向上的梯度。  
                            这两个卷积核的系数比Sobel算子更加精细，从而能够提供更加准确的边缘检测结果。  
                            $$
                            G_x = \\begin{bmatrix}
                            -3 & 0 & 3 \\\\
                            -10 & 0 & 10 \\\\
                            -3 & 0 & 3
                            \\end{bmatrix}
                            $$
                            $$
                            G_y = \\begin{bmatrix}
                            -3 & -10 & -3 \\\\
                            0 & 0 & 0 \\\\
                            3 & 10 & 3
                            \\end{bmatrix}
                            $$
                            **这里值提供了基础参数使用**
                            """)
                    with gr.Row():
                        x_alpha = gr.Slider(0, 1, step=0.01, value=0.5, label="x_alpha", info="x混合比例")
                        y_alpha = gr.Slider(0, 1, step=0.01, value=0.5, label="y_alpha", info="y混合比例")
                        scharr_btn = gr.Button(scale=1, value="边缘检测:scharr")
                        scharr_btn.click(fn=handle_wrapper(cv2_scharr), inputs=[show_image, x_alpha, y_alpha], outputs=[show_image, his_textbox])
                if handel_code == 21:
                    gr.Markdown("""
                            ## 边缘检测:laplacian
                            Laplacian算子是一种二阶微分算子，它通过对图像进行二次微分操作来检测边缘。  
                            Laplacian算子能够捕捉到图像中灰度变化的二阶导数峰值，从而定位边缘。  
                            然而，Laplacian算子对噪声敏感，因为它会增强图像中的高频部分，包括噪声。  
                            $$
                            \\begin{bmatrix}
                            0 & 1 & 0 \\\\
                            1 & -4 & 1 \\\\
                            0 & 1 & 0
                            \\end{bmatrix}
                            $$
                            """)
                    with gr.Row():
                        kernel_size = gr.Number(minimum=3, label="kernel_size", step=2, info="算子核的大小，正奇数。", value=3)
                        laplacian_btn = gr.Button(scale=1, value="边缘检测:laplacian")
                        laplacian_btn.click(fn=handle_wrapper(cv2_laplacian), inputs=[show_image, kernel_size], outputs=[show_image, his_textbox])
                if handel_code == 22:
                    gr.Markdown("""
                            ## 边缘检测:Canny
                            Canny边缘检测算法通常包括以下几个步骤：  
                            1. **灰度化处理**：将输入的彩色图像转换为灰度图像，以便后续处理。  
                            2. **高斯滤波**：使用高斯滤波器对灰度图像进行平滑处理，以减少噪声的影响。高斯滤波器的选择对边缘检测的效果有重要影响，一般选择5x5的核大小较为合适。  
                            3. **计算图像梯度**：使用Sobel算子或其他梯度算子计算图像在水平和垂直方向上的梯度值，从而得到每个像素点的梯度强度和方向。  
                            4. **非极大值抑制**：通过比较每个像素点在其梯度方向上的邻近像素点的梯度值，只保留梯度方向上的局部最大值点，以消除边缘检测带来的杂散响应。  
                            5. **双阈值处理**：设置两个阈值（高阈值和低阈值），将像素点的梯度值与这两个阈值进行比较，从而将像素点分为强边缘、弱边缘和非边缘三类。强边缘点直接保留为边缘点，非边缘点则被抑制，而弱边缘点则需要进一步处理。  
                            6. **边缘连接**：通过查看弱边缘点及其邻域内的强边缘点，将弱边缘点与强边缘点连接起来，形成完整的边缘。这一步有助于消除由噪声或颜色变化引起的孤立边缘点。  
                            """)
                    with gr.Row():
                        threshold1 = gr.Slider(0, 255, step=0.1, value=100, label="低阈值", info="非边缘点则被抑制")
                        threshold2 = gr.Slider(0, 255, step=0.1, value=200, label="高阈值", info="强边缘点直接保留为边缘点")
                        canny_btn = gr.Button(scale=1, value="边缘检测:Canny")
                        canny_btn.click(fn=handle_wrapper(cv2_canny), inputs=[show_image, threshold1, threshold2], outputs=[show_image, his_textbox])
                if handel_code == 23:
                    gr.Markdown("""
                            ## 轮廓检测:Contours
                            findContours函数在OpenCV中基于图像预处理、边缘检测（可选）、轮廓查找算法和轮廓近似算法的组合来查找图像中的轮廓。
                            此方法需要使用单通道的灰度图像,如果传入的是非灰度图,则会执行转换后处理
                            """)
                    with gr.Column():
                        contours_mode = gr.Dropdown(scale=1, label="轮廓检索模式", value="cv.RETR_EXTERNAL", choices=[
                            ("只检索最外层的轮廓", "cv.RETR_EXTERNAL"),
                            ("检索所有的轮廓，但不建立层次结构", "cv.RETR_LIST"),
                            ("检索所有的轮廓，并将轮廓组织为两层：外层轮廓和它们的孔", "cv.RETR_CCOMP"),
                            ("检索所有的轮廓，并重构轮廓的完整层次结构", "cv.RETR_TREE"),
                        ])
                        contours_method = gr.Dropdown(scale=1, label="轮廓近似方法", value="cv.CHAIN_APPROX_SIMPLE", choices=[
                            ("存储轮廓上所有的点，不进行任何近似", "cv.CHAIN_APPROX_NONE"),
                            ("仅存储轮廓的拐点，这可以显著减少点的数量，同时保持轮廓的形状", "cv.CHAIN_APPROX_SIMPLE"),
                            ("CV_CHAIN_APPROX_TC89系列的轮廓近似方法", "cv.CHAIN_APPROX_TC89_L1"),
                        ])
                        contours_btn = gr.Button(scale=0, value="轮廓检测:contours")
                        console_text = gr.Textbox(scale=5, label="边缘检测结果", value="输出内容")
                        contours_btn.click(fn=handle_wrapper(cv2_contours), inputs=[show_image, contours_mode, contours_method],
                                           outputs=[show_image, console_text, his_textbox])
                if handel_code == 24:
                    gr.Markdown("施工中(未完成)......")
demo = gr.TabbedInterface([cv_playground],
                          ["cv_playground"])

if __name__ == "__main__":
    demo.launch(server_name="[::]",server_port=64121)
