from manim import *

class ReluActivationFunction(Scene):
    def construct(self):
        # Set a 3blue1brown-style color scheme
        self.camera.background_color = "#333333"
        TEXT_COLOR = WHITE
        NEURON_COLOR = BLUE
        ACTIVE_NEURON_COLOR = GREEN
        DEACTIVATED_NEURON_COLOR = GRAY
        GRADIENT_COLOR = YELLOW

        # I. Introduction to Non-Linearity (0:00 - 0:45)
        
        # 1. Linear Limitation
        title = Text("The Problem with Linearity", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(title))
        self.wait(1)

        # Create a simple 2-layer neural network
        layer1_neurons = 3
        layer2_neurons = 2
        
        layer1 = VGroup(*[Circle(radius=0.5, color=NEURON_COLOR, fill_opacity=0.5) for _ in range(layer1_neurons)]).arrange(DOWN, buff=1)
        layer2 = VGroup(*[Circle(radius=0.5, color=NEURON_COLOR, fill_opacity=0.5) for _ in range(layer2_neurons)]).arrange(DOWN, buff=1.5)
        
        network = VGroup(layer1, layer2).arrange(RIGHT, buff=4)
        self.play(FadeIn(network))

        # Draw connections between layers
        connections = VGroup()
        for n1 in layer1:
            for n2 in layer2:
                connections.add(Line(n1.get_center(), n2.get_center(), stroke_width=2, color=WHITE))
        self.play(ShowCreation(connections))
        self.wait(1)

        # Show that stacking linear functions is still linear
        linear_function_1 = MathTex("y = w_1x + b_1", color=TEXT_COLOR).next_to(layer1, LEFT)
        linear_function_2 = MathTex("z = w_2y + b_2", color=TEXT_COLOR).next_to(layer2, RIGHT)
        self.play(Write(linear_function_1), Write(linear_function_2))
        self.wait(2)

        combined_function = MathTex("z = w_2(w_1x + b_1) + b_2", color=TEXT_COLOR).next_to(network, DOWN, buff=1)
        simplified_function = MathTex("z = (w_2w_1)x + (w_2b_1 + b_2)", color=TEXT_COLOR).next_to(combined_function, DOWN)
        still_linear = Text("Still a linear function!", color=RED).next_to(simplified_function, DOWN)
        self.play(Transform(VGroup(linear_function_1.copy(), linear_function_2.copy()), combined_function))
        self.wait(1)
        self.play(Transform(combined_function, simplified_function))
        self.play(Write(still_linear))
        self.wait(2)
        
        self.play(FadeOut(VGroup(title, network, connections, linear_function_1, linear_function_2, combined_function, still_linear)))
        
        # 2. The Problem with non-linear data
        title = Text("The Need for Non-Linearity", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(title))

        # Create non-linear data (concentric circles)
        circle1_dots = VGroup(*[Dot(point=2*np.array([np.cos(theta), np.sin(theta), 0]), color=BLUE) for theta in np.linspace(0, 2*PI, 20)])
        circle2_dots = VGroup(*[Dot(point=3*np.array([np.cos(theta), np.sin(theta), 0]), color=RED) for theta in np.linspace(0, 2*PI, 20)])
        self.play(ShowCreation(circle1_dots), ShowCreation(circle2_dots))

        # Show a linear model failing to separate the data
        line = Line(5*LEFT, 5*RIGHT, color=WHITE)
        self.play(ShowCreation(line))
        self.wait(1)
        self.play(Rotate(line, PI/2))
        self.wait(1)
        self.play(Rotate(line, PI/4))
        
        fail_text = Text("A linear model cannot separate this data.", color=TEXT_COLOR).next_to(line, DOWN)
        self.play(Write(fail_text))
        self.wait(2)

        self.play(FadeOut(VGroup(circle1_dots, circle2_dots, line, fail_text)))

        # 3. The Solution: Non-Linear Activation Function
        solution_text = Text("We need a non-linear activation function!", color=GREEN).scale(1.2)
        self.play(Write(solution_text))
        self.wait(2)
        self.play(FadeOut(solution_text), FadeOut(title))


        # II. ReLU: Definition and Core Function (0:45 - 2:00)

        # 1. The Formula & Graph
        relu_title = Text("ReLU: Rectified Linear Unit", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(relu_title))

        # Create axes
        axes = Axes(
            x_range=[-5, 5, 1],
            y_range=[-2, 5, 1],
            axis_config={"color": BLUE},
        )
        labels = axes.get_axis_labels(x_label="x", y_label="R(x)")
        self.play(Create(axes), Write(labels))

        # ReLU function
        relu_graph = axes.get_graph(lambda x: max(0, x), color=GREEN)
        relu_label = axes.get_graph_label(relu_graph, "\\max(0, x)").next_to(relu_graph, UR)
        self.play(ShowCreation(relu_graph), Write(relu_label))
        self.wait(2)
        
        # 2. Input Flow Visual
        
        # Positive Input
        dot_pos = Dot(axes.c2p(3, 0), color=YELLOW)
        line_pos_in = DashedLine(dot_pos.get_center(), axes.c2p(3, 3))
        dot_pos_out = Dot(axes.c2p(3, 3), color=YELLOW)
        line_pos_out = DashedLine(dot_pos_out.get_center(), axes.c2p(5, 3))
        
        self.play(FadeIn(dot_pos))
        self.play(ShowCreation(line_pos_in))
        self.play(Transform(dot_pos, dot_pos_out))
        self.play(ShowCreation(line_pos_out))
        self.wait(1)
        self.play(FadeOut(VGroup(dot_pos, line_pos_in, line_pos_out)))
        
        # Negative Input
        dot_neg = Dot(axes.c2p(-2, 0), color=RED)
        line_neg_in = DashedLine(dot_neg.get_center(), axes.c2p(-2, 0))
        dot_neg_out = Dot(axes.c2p(-2, 0), color=RED)
        line_neg_out = DashedLine(dot_neg_out.get_center(), axes.c2p(5, 0))

        self.play(FadeIn(dot_neg))
        self.play(ShowCreation(line_neg_in))
        self.play(Transform(dot_neg, dot_neg_out))
        self.play(ShowCreation(line_neg_out))
        self.wait(1)
        self.play(FadeOut(VGroup(dot_neg, line_neg_in, line_neg_out)))
        
        # Zero Input
        dot_zero = Dot(axes.c2p(0, 0), color=WHITE)
        line_zero_out = DashedLine(dot_zero.get_center(), axes.c2p(5, 0))

        self.play(FadeIn(dot_zero))
        self.play(ShowCreation(line_zero_out))
        self.wait(1)
        self.play(FadeOut(VGroup(dot_zero, line_zero_out)))
        
        # 3. Range Contrast
        input_range_text = MathTex("Input: (-\\infty, \\infty)", color=TEXT_COLOR).to_edge(DL)
        output_range_text = MathTex("Output: [0, \\infty)", color=GREEN).to_edge(DR)
        self.play(Write(input_range_text), Write(output_range_text))
        self.wait(2)
        self.play(FadeOut(VGroup(axes, labels, relu_graph, relu_label, input_range_text, output_range_text, relu_title)))
        

        # III. Advantages: Efficiency and Sparsity (2:00 - 3:00)
        
        # 1. Computational Efficiency
        adv_title = Text("Advantages of ReLU", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(adv_title))

        relu_eq = MathTex("ReLU(x) = \\max(0, x)", color=GREEN)
        sigmoid_eq = MathTex("Sigmoid(x) = \\frac{1}{1 + e^{-x}}", color=ORANGE)
        
        eqs = VGroup(relu_eq, sigmoid_eq).arrange(DOWN, buff=1).center()
        self.play(Write(relu_eq))
        self.wait(1)
        self.play(Write(sigmoid_eq))
        
        efficiency_text = Text("Simpler and faster computation", color=TEXT_COLOR).next_to(relu_eq, RIGHT)
        self.play(Write(efficiency_text))
        self.wait(2)
        
        self.play(FadeOut(VGroup(relu_eq, sigmoid_eq, efficiency_text)))
        
        # 2. Sparsity Visualization
        sparsity_title = Text("Sparsity", color=TEXT_COLOR).next_to(adv_title, DOWN)
        self.play(Write(sparsity_title))
        
        # Create a layer of 7 neurons
        neuron_layer = VGroup(*[Circle(radius=0.5, color=NEURON_COLOR, fill_opacity=0.5) for _ in range(7)]).arrange(RIGHT, buff=0.5)
        self.play(FadeIn(neuron_layer))
        
        # Apply ReLU
        relu_text = Text("Apply ReLU", color=TEXT_COLOR).next_to(neuron_layer, UP, buff=1)
        self.play(Write(relu_text))
        
        # Deactivate some neurons
        deactivated_neurons = VGroup(neuron_layer[1], neuron_layer[3], neuron_layer[4], neuron_layer[6])
        self.play(deactivated_neurons.animate.set_color(DEACTIVATED_NEURON_COLOR))

        sparsity_counter = Text("4/7 neurons deactivated", color=TEXT_COLOR).next_to(neuron_layer, DOWN, buff=1)
        self.play(Write(sparsity_counter))
        self.wait(2)
        self.play(FadeOut(VGroup(adv_title, sparsity_title, neuron_layer, relu_text, sparsity_counter)))

        
        # IV. Backpropagation and the Gradient (3:00 - 4:15)

        # 1. The Derivative
        bp_title = Text("Backpropagation with ReLU", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(bp_title))
        
        axes = Axes(x_range=[-5, 5, 1], y_range=[-2, 5, 1], axis_config={"color": BLUE})
        labels = axes.get_axis_labels(x_label="x", y_label="R(x)")
        relu_graph = axes.get_graph(lambda x: max(0, x), color=GREEN)
        self.play(Create(axes), Write(labels), ShowCreation(relu_graph))

        derivative_text = MathTex("\\frac{dR}{dx}", color=GRADIENT_COLOR).next_to(axes, UR)
        self.play(Write(derivative_text))

        # 2. Active Case (x > 0)
        active_segment = axes.get_graph(lambda x: x, x_range=[0.01, 5], color=YELLOW)
        self.play(ShowCreation(active_segment))
        
        slope_1_text = MathTex("\\frac{dR}{dx} = 1", color=GRADIENT_COLOR).next_to(active_segment, UR)
        self.play(Write(slope_1_text))
        self.wait(2)
        
        # Animate gradient flow
        gradient_in = Arrow(start=RIGHT, end=LEFT, color=GRADIENT_COLOR).next_to(slope_1_text, RIGHT)
        gradient_out = Arrow(start=RIGHT, end=LEFT, color=GRADIENT_COLOR).next_to(gradient_in, LEFT*5)
        self.play(ShowCreation(gradient_in))
        self.play(Transform(gradient_in, gradient_out))
        
        chain_rule_1 = MathTex("\\frac{\\partial L}{\\partial x} = 1 \\cdot \\frac{\\partial L}{\\partial y}", color=TEXT_COLOR).next_to(gradient_in, DOWN)
        self.play(Write(chain_rule_1))
        self.wait(2)
        
        self.play(FadeOut(VGroup(active_segment, slope_1_text, gradient_in, chain_rule_1)))
        
        # 3. Inactive Case (x <= 0)
        inactive_segment = axes.get_graph(lambda x: 0, x_range=[-5, 0], color=RED)
        self.play(ShowCreation(inactive_segment))
        
        slope_0_text = MathTex("\\frac{dR}{dx} = 0", color=GRADIENT_COLOR).next_to(inactive_segment, UL)
        self.play(Write(slope_0_text))
        self.wait(2)
        
        # Animate gradient annihilation
        gradient_in_2 = Arrow(start=RIGHT, end=LEFT, color=GRADIENT_COLOR).next_to(slope_0_text, RIGHT)
        self.play(ShowCreation(gradient_in_2))
        self.play(FadeOut(gradient_in_2, scale=0))
        
        chain_rule_0 = MathTex("\\frac{\\partial L}{\\partial x} = 0 \\cdot \\frac{\\partial L}{\\partial y} = 0", color=TEXT_COLOR).next_to(gradient_in_2, DOWN)
        self.play(Write(chain_rule_0))
        self.wait(2)

        self.play(FadeOut(VGroup(inactive_segment, slope_0_text, chain_rule_0, axes, labels, relu_graph, derivative_text, bp_title)))
        
        # 4. Vanishing Gradient
        vg_title = Text("Mitigating the Vanishing Gradient Problem", color=TEXT_COLOR).to_edge(UP)
        self.play(Write(vg_title))
        
        # ReLU gradient
        relu_grad_text = Text("ReLU Gradient", color=GREEN).shift(UP*2)
        relu_grad_values = MathTex("1, 1, 1, ...", color=TEXT_COLOR).next_to(relu_grad_text, DOWN)
        
        # Sigmoid gradient
        sigmoid_grad_text = Text("Sigmoid Gradient", color=ORANGE).shift(DOWN*2)
        sigmoid_grad_values = MathTex("0.2, 0.1, 0.05, ...", color=TEXT_COLOR).next_to(sigmoid_grad_text, DOWN)
        
        self.play(Write(relu_grad_text), Write(relu_grad_values))
        self.play(Write(sigmoid_grad_text), Write(sigmoid_grad_values))
        self.wait(2)

        self.play(FadeOut(VGroup(vg_title, relu_grad_text, relu_grad_values, sigmoid_grad_text, sigmoid_grad_values)))


        # V. The "Dying ReLU" Problem (4:15 - 5:00)

        # 1. The Setup
        dying_title = Text("The 'Dying ReLU' Problem", color=RED).to_edge(UP)
        self.play(Write(dying_title))

        neuron = Circle(radius=0.5, color=ACTIVE_NEURON_COLOR, fill_opacity=0.5).shift(LEFT*3)
        neuron_label = MathTex("x > 0", color=TEXT_COLOR).next_to(neuron, DOWN)
        self.play(FadeIn(neuron), Write(neuron_label))
        
        # 2. The Catastrophe
        gradient = Arrow(start=RIGHT*2, end=LEFT*2, color=RED, buff=0).next_to(neuron, RIGHT)
        gradient_label = MathTex("-ve \\text{ gradient}", color=RED).next_to(gradient, UP)
        self.play(ShowCreation(gradient), Write(gradient_label))
        
        update_text = Text("Large weight update", color=TEXT_COLOR).next_to(gradient, DOWN)
        self.play(Write(update_text))
        self.wait(2)
        
        # 3. The Death
        self.play(neuron.animate.set_color(DEACTIVATED_NEURON_COLOR), Transform(neuron_label, MathTex("x' < 0", color=TEXT_COLOR).next_to(neuron, DOWN)))
        self.play(FadeOut(gradient), FadeOut(gradient_label), FadeOut(update_text))
        
        permanent_text = Text("Permanently deactivated", color=RED).next_to(neuron, RIGHT)
        self.play(Write(permanent_text))
        self.wait(2)
        
        # 4. Final Visual
        gradient_zero_text = MathTex("\\frac{\\partial L}{\\partial x} = 0", color=TEXT_COLOR).next_to(permanent_text, DOWN)
        self.play(Write(gradient_zero_text))
        
        no_learning_text = Text("Cannot learn anymore!", color=RED).next_to(gradient_zero_text, DOWN)
        self.play(Write(no_learning_text))
        self.wait(3)

        self.play(FadeOut(VGroup(dying_title, neuron, neuron_label, permanent_text, gradient_zero_text, no_learning_text)))
        
        # 5. Conclusion
        conclusion_title = Text("Solutions", color=GREEN).to_edge(UP)
        self.play(Write(conclusion_title))
        
        leaky_relu_text = Text("Leaky ReLU", color=TEXT_COLOR).center()
        self.play(Write(leaky_relu_text))
        
        leaky_relu_eq = MathTex("f(x) = \\max(0.01x, x)", color=TEXT_COLOR).next_to(leaky_relu_text, DOWN)
        self.play(Write(leaky_relu_eq))
        
        axes = Axes(x_range=[-5, 5, 1], y_range=[-2, 5, 1], axis_config={"color": BLUE}).scale(0.5).next_to(leaky_relu_eq, DOWN)
        leaky_relu_graph = axes.get_graph(lambda x: max(0.01*x, x), color=GREEN)
        self.play(Create(axes), ShowCreation(leaky_relu_graph))
        self.wait(3)