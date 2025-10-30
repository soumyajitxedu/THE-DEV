"""
ReLU Activation Function - Educational Animation
A 3Blue1Brown-style visualization explaining the Rectified Linear Unit

Author: Generated for Educational Purposes
Duration: ~5 minutes
Manim Community Edition v0.17.0+
"""

from manim import *
import numpy as np

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

# Color scheme
RELU_BLUE = "#58C4DD"
RELU_RED = "#FF6B6B"
GRADIENT_GREEN = "#4ECDC4"
DEAD_NEURON_GRAY = "#555555"
ACTIVE_YELLOW = "#FFE66D"


# ============================================================================
# HELPER CLASSES
# ============================================================================

class NeuronLayer(VGroup):
    """Represents a layer of neurons in a neural network"""
    def __init__(self, n_neurons=5, **kwargs):
        super().__init__(**kwargs)
        self.neurons = VGroup(*[
            Circle(radius=0.3, fill_opacity=0.8, fill_color=BLUE)
            for _ in range(n_neurons)
        ])
        self.neurons.arrange(DOWN, buff=0.5)
        self.add(self.neurons)
    
    def activate_neurons(self, active_indices):
        """Highlight specific neurons as active"""
        anims = []
        for i, neuron in enumerate(self.neurons):
            if i in active_indices:
                anims.append(neuron.animate.set_fill(ACTIVE_YELLOW, 1.0))
            else:
                anims.append(neuron.animate.set_fill(DEAD_NEURON_GRAY, 0.3))
        return anims


class SimpleNetwork(VGroup):
    """A simple neural network visualization"""
    def __init__(self, layer_sizes=[3, 4, 3], **kwargs):
        super().__init__(**kwargs)
        self.layers = VGroup()
        
        for size in layer_sizes:
            layer = NeuronLayer(size)
            self.layers.add(layer)
        
        self.layers.arrange(RIGHT, buff=2.0)
        
        # Create connections
        self.connections = VGroup()
        for i in range(len(self.layers) - 1):
            layer1 = self.layers[i].neurons
            layer2 = self.layers[i + 1].neurons
            
            for n1 in layer1:
                for n2 in layer2:
                    line = Line(
                        n1.get_center(), n2.get_center(),
                        stroke_width=1, stroke_opacity=0.3,
                        color=GRAY
                    )
                    self.connections.add(line)
        
        self.add(self.connections, self.layers)


# ============================================================================
# SECTION I: INTRODUCTION TO NON-LINEARITY (0:00 - 0:45)
# ============================================================================

class IntroNonLinearity(Scene):
    def construct(self):
        # Title
        title = Text("The Need for Non-Linearity", font_size=48)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(2)
        
        # Create a simple 2-layer network
        network = SimpleNetwork([3, 4, 2])
        network.scale(0.7)
        network.shift(LEFT * 2)
        
        self.play(FadeIn(network, shift=RIGHT))
        self.wait(1)
        
        # Show linear transformation equations
        linear1 = MathTex(r"f_1(x) = W_1 x + b_1", font_size=36)
        linear2 = MathTex(r"f_2(x) = W_2 x + b_2", font_size=36)
        composition = MathTex(
            r"f_2(f_1(x)) = W_2(W_1 x + b_1) + b_2",
            font_size=36
        )
        simplified = MathTex(
            r"= (W_2 W_1) x + (W_2 b_1 + b_2)",
            font_size=36
        )
        final = MathTex(r"= W' x + b'", font_size=36, color=RELU_RED)
        
        equations = VGroup(linear1, linear2, composition, simplified, final)
        equations.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        equations.next_to(network, RIGHT, buff=1.0)
        
        self.play(Write(linear1))
        self.wait(1)
        self.play(Write(linear2))
        self.wait(1)
        self.play(Write(composition))
        self.wait(2)
        self.play(Write(simplified))
        self.wait(1)
        self.play(Write(final))
        self.wait(2)
        
        # Problem statement
        problem = Text(
            "Multiple linear layers\n= Just one linear layer!",
            font_size=32,
            color=RELU_RED
        )
        problem.next_to(equations, DOWN, buff=0.5)
        self.play(FadeIn(problem, scale=1.2))
        self.wait(2)
        
        # Clear and show non-linear data
        self.play(
            FadeOut(network),
            FadeOut(equations),
            FadeOut(problem),
            FadeOut(title)
        )
        
        # Create spiral data visualization
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": False}
        )
        axes.shift(LEFT * 3)
        
        # Generate spiral data
        theta = np.linspace(0, 4 * np.pi, 100)
        r = np.linspace(0.5, 2.5, 100)
        
        blue_points = VGroup(*[
            Dot(axes.c2p(r[i] * np.cos(theta[i]), r[i] * np.sin(theta[i])),
                radius=0.05, color=BLUE)
            for i in range(0, 100, 2)
        ])
        
        red_points = VGroup(*[
            Dot(axes.c2p(r[i] * np.cos(theta[i] + np.pi), 
                        r[i] * np.sin(theta[i] + np.pi)),
                radius=0.05, color=RED)
            for i in range(0, 100, 2)
        ])
        
        data_label = Text("Non-linear Data", font_size=36)
        data_label.next_to(axes, UP)
        
        self.play(
            Create(axes),
            Write(data_label)
        )
        self.play(
            LaggedStart(*[FadeIn(d) for d in blue_points], lag_ratio=0.01),
            LaggedStart(*[FadeIn(d) for d in red_points], lag_ratio=0.01),
            run_time=2
        )
        self.wait(1)
        
        # Show linear boundary failing
        line = Line(axes.c2p(-3, -2), axes.c2p(3, 2), color=YELLOW, stroke_width=3)
        fail_label = Text("Linear boundary fails!", font_size=32, color=RELU_RED)
        fail_label.next_to(axes, DOWN)
        
        self.play(Create(line))
        self.play(Write(fail_label))
        self.wait(2)
        
        # Solution
        solution = Text(
            "Solution: Non-Linear Activation Functions",
            font_size=40,
            color=GRADIENT_GREEN
        )
        solution.to_edge(RIGHT).shift(RIGHT * 0.5)
        
        self.play(Write(solution))
        self.wait(2)
        
        # Transition
        self.play(
            *[FadeOut(mob) for mob in self.mobjects]
        )


# ============================================================================
# SECTION II: RELU DEFINITION AND CORE FUNCTION (0:45 - 2:00)
# ============================================================================

class ReLUDefinition(Scene):
    def construct(self):
        # Title
        title = Text("ReLU: Rectified Linear Unit", font_size=52)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show the formula
        formula = MathTex(
            r"R(x) = \max(0, x)",
            font_size=72
        )
        formula.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(formula))
        self.wait(2)
        
        # Alternative representation
        alt_formula = MathTex(
            r"R(x) = \begin{cases} x & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}",
            font_size=48
        )
        alt_formula.next_to(formula, DOWN, buff=0.5)
        
        self.play(Write(alt_formula))
        self.wait(2)
        
        # Clear formulas
        self.play(
            FadeOut(formula),
            FadeOut(alt_formula)
        )
        
        # Create axes for ReLU graph
        axes = Axes(
            x_range=[-4, 4, 1],
            y_range=[-1, 4, 1],
            x_length=8,
            y_length=6,
            axis_config={
                "include_tip": True,
                "numbers_to_include": range(-4, 5)
            }
        )
        axes.center().shift(DOWN * 0.5)
        
        # Axis labels
        x_label = MathTex("x", font_size=36).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("R(x)", font_size=36).next_to(axes.y_axis.get_end(), UP)
        
        self.play(
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        self.wait(1)
        
        # Draw ReLU function - negative part first
        relu_negative = Line(
            axes.c2p(-4, 0),
            axes.c2p(0, 0),
            color=RELU_RED,
            stroke_width=4
        )
        
        # Positive part
        relu_positive = Line(
            axes.c2p(0, 0),
            axes.c2p(4, 4),
            color=RELU_BLUE,
            stroke_width=4
        )
        
        # Labels for each part
        neg_label = MathTex(r"y = 0", font_size=32, color=RELU_RED)
        neg_label.next_to(axes.c2p(-2, 0), UP)
        
        pos_label = MathTex(r"y = x", font_size=32, color=RELU_BLUE)
        pos_label.next_to(axes.c2p(2, 2), UP + LEFT)
        
        # Animate drawing the function
        self.play(Create(relu_negative), Write(neg_label))
        self.wait(1)
        self.play(Create(relu_positive), Write(pos_label))
        self.wait(1)
        
        # Highlight the kink at x=0
        kink_dot = Dot(axes.c2p(0, 0), color=YELLOW, radius=0.1)
        kink_label = Text("Kink at x=0", font_size=28, color=YELLOW)
        kink_label.next_to(kink_dot, DOWN + RIGHT)
        
        self.play(
            FadeIn(kink_dot, scale=2),
            Write(kink_label)
        )
        self.wait(2)
        
        # Input flow demonstration
        self.play(
            FadeOut(neg_label),
            FadeOut(pos_label),
            FadeOut(kink_label),
            FadeOut(kink_dot)
        )
        
        # Test input 1: Positive
        input1_value = 3
        input1_dot = Dot(axes.c2p(input1_value, 0), color=GREEN, radius=0.08)
        input1_arrow = Arrow(
            axes.c2p(input1_value, -0.5),
            input1_dot.get_center(),
            color=GREEN,
            buff=0
        )
        input1_label = MathTex(f"x = {input1_value}", font_size=32, color=GREEN)
        input1_label.next_to(input1_arrow, DOWN)
        
        self.play(
            GrowArrow(input1_arrow),
            FadeIn(input1_dot),
            Write(input1_label)
        )
        self.wait(0.5)
        
        # Trace to output
        v_line1 = DashedLine(
            input1_dot.get_center(),
            axes.c2p(input1_value, input1_value),
            color=GREEN,
            dash_length=0.1
        )
        output1_dot = Dot(axes.c2p(input1_value, input1_value), color=GREEN, radius=0.08)
        output1_label = MathTex(f"R(x) = {input1_value}", font_size=32, color=GREEN)
        output1_label.next_to(output1_dot, RIGHT)
        
        self.play(Create(v_line1))
        self.play(
            FadeIn(output1_dot, scale=2),
            Write(output1_label)
        )
        self.wait(1.5)
        
        # Clear first example
        self.play(
            *[FadeOut(mob) for mob in [
                input1_arrow, input1_dot, input1_label,
                v_line1, output1_dot, output1_label
            ]]
        )
        
        # Test input 2: Negative
        input2_value = -2
        input2_dot = Dot(axes.c2p(input2_value, 0), color=RED, radius=0.08)
        input2_arrow = Arrow(
            axes.c2p(input2_value, -0.5),
            input2_dot.get_center(),
            color=RED,
            buff=0
        )
        input2_label = MathTex(f"x = {input2_value}", font_size=32, color=RED)
        input2_label.next_to(input2_arrow, DOWN)
        
        self.play(
            GrowArrow(input2_arrow),
            FadeIn(input2_dot),
            Write(input2_label)
        )
        self.wait(0.5)
        
        # Show clipping to zero
        clip_arrow = Arrow(
            input2_dot.get_center(),
            axes.c2p(0, 0),
            color=RED,
            buff=0.1
        )
        output2_dot = Dot(axes.c2p(0, 0), color=RED, radius=0.08)
        output2_label = MathTex("R(x) = 0", font_size=32, color=RED)
        output2_label.next_to(axes.c2p(0, 0), UP + RIGHT)
        clip_text = Text("Clipped!", font_size=28, color=YELLOW)
        clip_text.next_to(clip_arrow, UP)
        
        self.play(GrowArrow(clip_arrow), Write(clip_text))
        self.play(
            FadeIn(output2_dot, scale=2),
            Write(output2_label)
        )
        self.wait(1.5)
        
        # Clear second example
        self.play(
            *[FadeOut(mob) for mob in [
                input2_arrow, input2_dot, input2_label,
                clip_arrow, clip_text, output2_dot, output2_label
            ]]
        )
        
        # Test input 3: Zero
        input3_value = 0
        input3_dot = Dot(axes.c2p(input3_value, 0), color=YELLOW, radius=0.08)
        input3_arrow = Arrow(
            axes.c2p(input3_value, -0.5),
            input3_dot.get_center(),
            color=YELLOW,
            buff=0
        )
        input3_label = MathTex("x = 0", font_size=32, color=YELLOW)
        input3_label.next_to(input3_arrow, DOWN + LEFT)
        
        output3_label = MathTex("R(x) = 0", font_size=32, color=YELLOW)
        output3_label.next_to(axes.c2p(0, 0), UP + LEFT)
        
        self.play(
            GrowArrow(input3_arrow),
            FadeIn(input3_dot),
            Write(input3_label)
        )
        self.wait(0.5)
        self.play(Write(output3_label))
        self.wait(1.5)
        
        # Range summary
        self.play(
            *[FadeOut(mob) for mob in [
                input3_arrow, input3_dot, input3_label, output3_label
            ]]
        )
        
        input_range = MathTex(
            r"\text{Input: } x \in (-\infty, \infty)",
            font_size=36
        )
        output_range = MathTex(
            r"\text{Output: } R(x) \in [0, \infty)",
            font_size=36,
            color=RELU_BLUE
        )
        
        ranges = VGroup(input_range, output_range)
        ranges.arrange(DOWN, buff=0.3)
        ranges.to_edge(RIGHT).shift(RIGHT * 0.5)
        
        self.play(Write(input_range))
        self.wait(1)
        self.play(Write(output_range))
        self.wait(2)
        
        # Transition
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# SECTION III: ADVANTAGES - EFFICIENCY AND SPARSITY (2:00 - 3:00)
# ============================================================================

class ReLUAdvantages(Scene):
    def construct(self):
        title = Text("ReLU Advantages", font_size=52)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Advantage 1: Computational Efficiency
        adv1_title = Text("1. Computational Efficiency", font_size=40, color=GRADIENT_GREEN)
        adv1_title.next_to(title, DOWN, buff=0.8)
        self.play(Write(adv1_title))
        self.wait(1)
        
        # Show ReLU computation
        relu_comp = VGroup(
            MathTex(r"\text{ReLU: } R(x) = \max(0, x)", font_size=36),
            Text("Simple comparison operation", font_size=28, color=GRAY)
        )
        relu_comp.arrange(DOWN, buff=0.2)
        relu_comp.shift(LEFT * 2 + DOWN * 1)
        
        # Show Sigmoid computation (for contrast)
        sigmoid_comp = VGroup(
            MathTex(r"\text{Sigmoid: } \sigma(x) = \frac{1}{1 + e^{-x}}", font_size=36),
            Text("Expensive exponential", font_size=28, color=GRAY)
        )
        sigmoid_comp.arrange(DOWN, buff=0.2)
        sigmoid_comp.shift(RIGHT * 2 + DOWN * 1)
        
        self.play(Write(relu_comp[0]))
        self.play(Write(relu_comp[1]))
        self.wait(1)
        self.play(Write(sigmoid_comp[0]))
        self.play(Write(sigmoid_comp[1]))
        self.wait(2)
        
        # Speed comparison
        speed_arrow = Arrow(relu_comp.get_right(), sigmoid_comp.get_left(), buff=0.3)
        speed_label = Text("Much Faster!", font_size=32, color=GREEN)
        speed_label.next_to(speed_arrow, UP)
        
        self.play(
            GrowArrow(speed_arrow),
            Write(speed_label)
        )
        self.wait(2)
        
        # Clear advantage 1
        self.play(
            *[FadeOut(mob) for mob in [
                adv1_title, relu_comp, sigmoid_comp,
                speed_arrow, speed_label
            ]]
        )
        
        # Advantage 2: Sparsity
        adv2_title = Text("2. Sparsity (Network Efficiency)", font_size=40, color=GRADIENT_GREEN)
        adv2_title.next_to(title, DOWN, buff=0.8)
        self.play(Write(adv2_title))
        self.wait(1)
        
        # Create a layer of neurons
        n_neurons = 7
        neurons = VGroup()
        for i in range(n_neurons):
            neuron = Circle(radius=0.4, fill_opacity=0.8, fill_color=BLUE, stroke_width=2)
            neurons.add(neuron)
        
        neurons.arrange(DOWN, buff=0.4)
        neurons.shift(LEFT * 3)
        
        # Create input values
        input_values = [2.5, -1.3, 3.7, -0.8, 1.2, -2.1, 0.5]
        input_labels = VGroup()
        for i, (neuron, val) in enumerate(zip(neurons, input_values)):
            label = MathTex(f"x_{{{i+1}}} = {val:.1f}", font_size=28)
            label.next_to(neuron, LEFT, buff=0.3)
            input_labels.add(label)
        
        self.play(
            LaggedStart(*[Create(n) for n in neurons], lag_ratio=0.1),
            LaggedStart(*[Write(l) for l in input_labels], lag_ratio=0.1),
            run_time=2
        )
        self.wait(1)
        
        # Apply ReLU
        relu_label = Text("Apply ReLU", font_size=36, color=RELU_BLUE)
        relu_label.next_to(neurons, RIGHT, buff=1.5)
        arrow = Arrow(neurons.get_right(), relu_label.get_left(), buff=0.3)
        
        self.play(
            GrowArrow(arrow),
            Write(relu_label)
        )
        self.wait(1)
        
        # Show activation/deactivation
        output_neurons = neurons.copy()
        output_neurons.shift(RIGHT * 6)
        
        output_values = [max(0, v) for v in input_values]
        output_labels = VGroup()
        
        activation_anims = []
        for i, (neuron, val, out_val) in enumerate(zip(output_neurons, input_values, output_values)):
            label = MathTex(f"R(x_{{{i+1}}}) = {out_val:.1f}", font_size=28)
            label.next_to(neuron, RIGHT, buff=0.3)
            output_labels.add(label)
            
            if val <= 0:
                # Inactive neuron
                activation_anims.append(
                    neuron.animate.set_fill(DEAD_NEURON_GRAY, 0.3)
                )
            else:
                # Active neuron
                activation_anims.append(
                    neuron.animate.set_fill(ACTIVE_YELLOW, 1.0)
                )
        
        self.play(
            TransformFromCopy(neurons, output_neurons),
            *activation_anims,
            run_time=2
        )
        self.play(
            LaggedStart(*[Write(l) for l in output_labels], lag_ratio=0.1),
            run_time=1.5
        )
        self.wait(2)
        
        # Count active neurons
        active_count = sum(1 for v in input_values if v > 0)
        inactive_count = n_neurons - active_count
        
        counter_text = Text(
            f"Active: {active_count}/{n_neurons}\nInactive: {inactive_count}/{n_neurons}",
            font_size=32,
            color=YELLOW
        )
        counter_text.to_edge(DOWN)
        
        sparse_label = Text(
            "Sparse Network → Better Performance!",
            font_size=36,
            color=GRADIENT_GREEN
        )
        sparse_label.next_to(counter_text, UP, buff=0.5)
        
        self.play(Write(counter_text))
        self.wait(1)
        self.play(Write(sparse_label))
        self.wait(2)
        
        # Transition
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# SECTION IV: BACKPROPAGATION AND GRADIENT (3:00 - 4:15)
# ============================================================================

class BackpropagationGradient(Scene):
    def construct(self):
        title = Text("Backpropagation Through ReLU", font_size=52)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Show derivative formula
        derivative_title = Text("The Derivative", font_size=40, color=GRADIENT_GREEN)
        derivative_title.next_to(title, DOWN, buff=0.6)
        
        derivative = MathTex(
            r"\frac{dR}{dx} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}",
            font_size=48
        )
        derivative.next_to(derivative_title, DOWN, buff=0.5)
        
        self.play(Write(derivative_title))
        self.wait(0.5)
        self.play(Write(derivative))
        self.wait(2)
        
        # Create ReLU graph
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-0.5, 3, 1],
            x_length=6,
            y_length=4,
            axis_config={"include_tip": True}
        )
        axes.shift(DOWN * 1 + LEFT * 3)
        
        x_label = MathTex("x", font_size=28).next_to(axes.x_axis.get_end(), RIGHT)
        y_label = MathTex("R(x)", font_size=28).next_to(axes.y_axis.get_end(), UP)
        
        relu_neg = Line(axes.c2p(-3, 0), axes.c2p(0, 0), color=RELU_RED, stroke_width=3)
        relu_pos = Line(axes.c2p(0, 0), axes.c2p(3, 3), color=RELU_BLUE, stroke_width=3)
        
        self.play(
            FadeOut(derivative_title),
            FadeOut(derivative),
            Create(axes),
            Write(x_label),
            Write(y_label)
        )
        self.play(Create(relu_neg), Create(relu_pos))
        self.wait(1)
        
        # CASE 1: Active case (x > 0)
        case1_title = Text("Case 1: Active (x > 0)", font_size=36, color=RELU_BLUE)
        case1_title.to_edge(RIGHT).shift(UP * 2)
        self.play(Write(case1_title))
        self.wait(1)
        
        # Highlight positive part
        highlight1 = relu_pos.copy().set_stroke(ACTIVE_YELLOW, width=8)
        self.play(AnimationGroup(Create(highlight1), Uncreate(highlight1), lag_ratio=1, run_time=1.5))
        
        # Show slope = 1
        slope_point = axes.c2p(1.5, 1.5)
        slope_line = Line(
            axes.c2p(1.0, 1.0),
            axes.c2p(2.0, 2.0),
            color=ACTIVE_YELLOW,
            stroke_width=4
        )
        slope_label = MathTex(r"\frac{dR}{dx} = 1", font_size=36, color=ACTIVE_YELLOW)
        slope_label.next_to(slope_line, RIGHT)
        
        self.play(Create(slope_line))
        self.play(Write(slope_label))
        self.wait(1)
        
        # Chain rule visualization
        chain_rule = MathTex(
            r"\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{dR}{dx}",
            font_size=32
        )
        chain_rule.next_to(case1_title, DOWN, buff=0.4)
        
        chain_rule_result = MathTex(
            r"= \frac{\partial L}{\partial y} \cdot 1 = \frac{\partial L}{\partial y}",
            font_size=32,
            color=GRADIENT_GREEN
        )
        chain_rule_result.next_to(chain_rule, DOWN, buff=0.3)
        
        self.play(Write(chain_rule))
        self.wait(1.5)
        self.play(Write(chain_rule_result))
        self.wait(1)
        
        # Gradient flow visualization
        grad_arrow = Arrow(
            axes.c2p(1.5, 2.5),
            axes.c2p(1.5, 1.5),
            color=GRADIENT_GREEN,
            buff=0.1,
            stroke_width=6
        )
        grad_label = MathTex(r"\frac{\partial L}{\partial y}", font_size=28, color=GRADIENT_GREEN)
        grad_label.next_to(grad_arrow, RIGHT)
        
        backgrad_arrow = Arrow(
            axes.c2p(1.5, 1.5),
            axes.c2p(0.5, 0.5),
            color=GRADIENT_GREEN,
            buff=0.1,
            stroke_width=6
        )
        backgrad_label = MathTex(r"\frac{\partial L}{\partial x}", font_size=28, color=GRADIENT_GREEN)
        backgrad_label.next_to(backgrad_arrow, DOWN)
        
        flow_text = Text("Gradient flows through!", font_size=28, color=GREEN)
        flow_text.next_to(chain_rule_result, DOWN, buff=0.3)
        
        self.play(GrowArrow(grad_arrow), Write(grad_label))
        self.wait(0.5)
        self.play(GrowArrow(backgrad_arrow), Write(backgrad_label))
        self.play(Write(flow_text))
        self.wait(2)
        
        # Clear case 1
        self.play(
            *[FadeOut(mob) for mob in [
                case1_title, slope_line, slope_label,
                chain_rule, chain_rule_result, flow_text,
                grad_arrow, grad_label, backgrad_arrow, backgrad_label
            ]]
        )
        
        # CASE 2: Inactive case (x <= 0)
        case2_title = Text("Case 2: Inactive (x ≤ 0)", font_size=36, color=RELU_RED)
        case2_title.to_edge(RIGHT).shift(UP * 2)
        self.play(Write(case2_title))
        self.wait(1)
        
        # Highlight negative part
        highlight2 = relu_neg.copy().set_stroke(RELU_RED, width=8)
        self.play(AnimationGroup(Create(highlight2), Uncreate(highlight2), lag_ratio=1, run_time=1.5))
        
        # Show slope = 0
        slope_point2 = axes.c2p(-1.5, 0)
        slope_line2 = Line(
            axes.c2p(-2.0, 0),
            axes.c2p(-1.0, 0),
            color=RELU_RED,
            stroke_width=4
        )
        slope_label2 = MathTex(r"\frac{dR}{dx} = 0", font_size=36, color=RELU_RED)
        slope_label2.next_to(slope_line2, UP)
        
        self.play(Create(slope_line2))
        self.play(Write(slope_label2))
        self.wait(1)
        
        # Chain rule for inactive case
        chain_rule2 = MathTex(
            r"\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{dR}{dx}",
            font_size=32
        )
        chain_rule2.next_to(case2_title, DOWN, buff=0.4)
        
        chain_rule_result2 = MathTex(
            r"= \frac{\partial L}{\partial y} \cdot 0 = 0",
            font_size=32,
            color=RELU_RED
        )
        chain_rule_result2.next_to(chain_rule2, DOWN, buff=0.3)
        
        self.play(Write(chain_rule2))
        self.wait(1.5)
        self.play(Write(chain_rule_result2))
        self.wait(1)
        
        # Gradient blocked visualization
        grad_arrow2 = Arrow(
            axes.c2p(-1.5, 1.5),
            axes.c2p(-1.5, 0.3),
            color=GRADIENT_GREEN,
            buff=0.1,
            stroke_width=6
        )
        
        block_cross = VGroup(
            Line(UL, DR, color=RED, stroke_width=6),
            Line(UR, DL, color=RED, stroke_width=6)
        ).scale(0.3)
        block_cross.move_to(axes.c2p(-1.5, 0))
        
        block_text = Text("Gradient blocked!", font_size=28, color=RED)
        block_text.next_to(chain_rule_result2, DOWN, buff=0.3)
        
        self.play(GrowArrow(grad_arrow2))
        self.wait(0.5)
        self.play(Create(block_cross))
        self.play(Write(block_text))
        self.wait(2)
        
        # Vanishing gradient comparison
        self.play(
            *[FadeOut(mob) for mob in [
                case2_title, slope_line2, slope_label2,
                chain_rule2, chain_rule_result2, block_text,
                grad_arrow2, block_cross
            ]]
        )
        
        comparison_title = Text(
            "ReLU vs. Sigmoid/Tanh",
            font_size=40,
            color=ACTIVE_YELLOW
        )
        comparison_title.to_edge(RIGHT).shift(UP * 2)
        
        relu_grad = MathTex(
            r"\text{ReLU: } \frac{dR}{dx} = 1 \text{ or } 0",
            font_size=32,
            color=GRADIENT_GREEN
        )
        relu_grad.next_to(comparison_title, DOWN, buff=0.5)
        
        sigmoid_grad = MathTex(
            r"\text{Sigmoid: } \frac{d\sigma}{dx} = \sigma(x)(1-\sigma(x))",
            font_size=32
        )
        sigmoid_grad.next_to(relu_grad, DOWN, buff=0.3)
        
        vanishing_note = Text(
            "ReLU mitigates\nvanishing gradients!",
            font_size=28,
            color=GREEN
        )
        vanishing_note.next_to(sigmoid_grad, DOWN, buff=0.5)
        
        self.play(Write(comparison_title))
        self.play(Write(relu_grad))
        self.wait(1)
        self.play(Write(sigmoid_grad))
        self.wait(1)
        self.play(Write(vanishing_note))
        self.wait(3)
        
        # Transition
        self.play(*[FadeOut(mob) for mob in self.mobjects])


# ============================================================================
# SECTION V: THE DYING RELU PROBLEM (4:15 - 5:00)
# ============================================================================

class DyingReLU(Scene):
    def construct(self):
        title = Text('The "Dying ReLU" Problem', font_size=52, color=RELU_RED)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait(1)
        
        # Create a single neuron
        neuron = Circle(radius=0.8, fill_opacity=0.8, fill_color=BLUE, stroke_width=3)
        neuron.shift(LEFT * 4)
        
        neuron_label = Text("Active Neuron", font_size=32)
        neuron_label.next_to(neuron, UP)
        
        # Initial positive input
        initial_value = MathTex("x = 2.5", font_size=36, color=GREEN)
        initial_value.next_to(neuron, DOWN)
        
        self.play(
            Create(neuron),
            Write(neuron_label),
            Write(initial_value)
        )
        self.wait(1)
        
        # Show ReLU output
        relu_box = Rectangle(width=2, height=1.5, color=RELU_BLUE, stroke_width=3)
        relu_box.shift(LEFT * 0.5)
        relu_text = Text("ReLU", font_size=28)
        relu_text.move_to(relu_box)
        
        arrow1 = Arrow(neuron.get_right(), relu_box.get_left(), buff=0.2)
        
        output_value = MathTex("R(x) = 2.5", font_size=32, color=GREEN)
        output_value.next_to(relu_box, RIGHT, buff=0.5)
        
        self.play(
            Create(relu_box),
            Write(relu_text),
            GrowArrow(arrow1)
        )
        self.play(Write(output_value))
        self.wait(2)
        
        # Training update scenario
        update_title = Text("During Training Update...", font_size=36, color=YELLOW)
        update_title.to_edge(RIGHT).shift(UP * 1.5)
        self.play(Write(update_title))
        self.wait(1)
        
        # Large negative gradient
        gradient = MathTex(
            r"\frac{\partial L}{\partial w} = -5.0",
            font_size=32,
            color=RED
        )
        gradient.next_to(update_title, DOWN, buff=0.4)
        
        weight_update = MathTex(
            r"w \leftarrow w - \alpha \frac{\partial L}{\partial w}",
            font_size=28
        )
        weight_update.next_to(gradient, DOWN, buff=0.3)
        
        self.play(Write(gradient))
        self.wait(1)
        self.play(Write(weight_update))
        self.wait(2)
        
        # Result: Shifted to negative
        catastrophe = Text("Catastrophe!", font_size=36, color=RED)
        catastrophe.next_to(weight_update, DOWN, buff=0.5)
        
        new_value = MathTex("x' = -1.8", font_size=36, color=RED)
        new_value.next_to(neuron, DOWN)
        
        self.play(Write(catastrophe))
        self.play(
            Transform(initial_value, new_value),
            neuron.animate.set_fill(DEAD_NEURON_GRAY, 0.5)
        )
        self.wait(2)
        
        # Show zero output
        new_output = MathTex("R(x') = 0", font_size=32, color=RED)
        new_output.next_to(relu_box, RIGHT, buff=0.5)
        
        self.play(Transform(output_value, new_output))
        self.wait(2)
        
        # Death visualization
        death_title = Text("The Neuron is Dead!", font_size=48, color=RED)
        death_title.next_to(title, DOWN, buff=0.8)
        
        self.play(Write(death_title))
        self.wait(1)
        
        # Show consequences
        consequences = VGroup(
            MathTex(r"\text{Output: } R(x') = 0", font_size=28),
            MathTex(r"\text{Gradient: } \frac{dR}{dx'} = 0", font_size=28),
            Text("→ No weight updates", font_size=28),
            Text("→ Neuron stuck forever!", font_size=28, color=RED)
        )
        consequences.arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        consequences.next_to(update_title, DOWN, buff=0.5)
        
        # Clear previous explanations
        self.play(
            FadeOut(gradient),
            FadeOut(weight_update),
            FadeOut(catastrophe)
        )
        
        for consequence in consequences:
            self.play(Write(consequence))
            self.wait(0.8)
        
        self.wait(2)
        
        # Visual emphasis on dead neuron
        dead_emphasis = neuron.copy()
        dead_emphasis.set_stroke(RED, width=6)
        cross = VGroup(
            Line(neuron.get_corner(UL), neuron.get_corner(DR), color=RED, stroke_width=8),
            Line(neuron.get_corner(UR), neuron.get_corner(DL), color=RED, stroke_width=8)
        )
        
        self.play(
            AnimationGroup(Create(dead_emphasis), Uncreate(dead_emphasis), lag_ratio=1, run_time=1.5),
            Create(cross)
        )
        self.wait(2)
        
        # Solutions teaser
        solutions_title = Text("Possible Solutions:", font_size=36, color=GRADIENT_GREEN)
        solutions_title.to_edge(DOWN).shift(UP * 1.5)
        
        solutions = VGroup(
            Text("• Leaky ReLU", font_size=28),
            Text("• Parametric ReLU (PReLU)", font_size=28),
            Text("• ELU (Exponential Linear Unit)", font_size=28)
        )
        solutions.arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        solutions.next_to(solutions_title, DOWN, buff=0.3)
        
        self.play(
            FadeOut(death_title),
            FadeOut(consequences),
            Write(solutions_title)
        )
        self.play(
            LaggedStart(*[Write(s) for s in solutions], lag_ratio=0.3),
            run_time=2
        )
        self.wait(3)
        
        # Final fade out
        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait(1)


# ============================================================================
# MAIN SCENE - COMPLETE 5-MINUTE VIDEO
# ============================================================================

class CompleteReLUVideo(Scene):
    """
    Main scene that combines all sections into a complete 5-minute video
    """
    def construct(self):
        # Section I: Introduction to Non-Linearity
        intro = IntroNonLinearity()
        intro.construct()
        
        # Section II: ReLU Definition and Core Function
        definition = ReLUDefinition()
        definition.construct()
        
        # Section III: Advantages - Efficiency and Sparsity
        advantages = ReLUAdvantages()
        advantages.construct()
        
        # Section IV: Backpropagation and Gradient
        backprop = BackpropagationGradient()
        backprop.construct()
        
        # Section V: The Dying ReLU Problem
        dying = DyingReLU()
        dying.construct()
        
        # Final title card
        final_title = Text("ReLU: Simple, Powerful, Essential", font_size=48)
        final_subtitle = Text(
            "The foundation of modern deep learning",
            font_size=32,
            color=GRAY
        )
        final_group = VGroup(final_title, final_subtitle)
        final_group.arrange(DOWN, buff=0.5)
        
        self.play(FadeIn(final_group, scale=1.2))
        self.wait(3)
        self.play(FadeOut(final_group))


# ============================================================================
# RENDERING INSTRUCTIONS
# ============================================================================

"""
To render this animation, use the following commands:

# Render the complete video (all sections):
manim -pqh relu_manim_animation.py CompleteReLUVideo

# Or render individual sections:
manim -pqh relu_manim_animation.py IntroNonLinearity
manim -pqh relu_manim_animation.py ReLUDefinition
manim -pqh relu_manim_animation.py ReLUAdvantages
manim -pqh relu_manim_animation.py BackpropagationGradient
manim -pqh relu_manim_animation.py DyingReLU

Flags:
-p: Preview after rendering
-q: Quality (l=low, m=medium, h=high)
-qh: High quality (1080p)

For 4K rendering:
manim -pqk relu_manim_animation.py CompleteReLUVideo

Total estimated runtime: ~5 minutes
"""