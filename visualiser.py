"""Visualizer for the Toy Environment."""
import os
import functools
import numpy as np
import Tkinter as tk
from typing import Optional
from absl import logging
from PIL import ImageTk as itk
from PIL import Image

# import synthetic_agents.toy_domain.env as env_lib
import np_mdp.utils.cache_utils as cache_lib


def wrapped_partial(func, *args, **kwargs):
  """Copies function properties to partial functions.
  Source: http://louistiao.me/posts/
  adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
  """
  partial_func = functools.partial(func, *args, **kwargs)
  functools.update_wrapper(partial_func, func)
  return partial_func


class ToyEnvVisualizer(tk.LabelFrame):
  """Visualizer for the toy environment."""

  def __init__(self,
               master,
               mdp_env,
               mdp_policy=None,
               start_state_idx=0):
    """Initializes the visualizer.
    Args:
      master: See tkinter for details.
      mdp_env: An MDP environment.
      mdp_policy: Optional. A stochastic policy specifying as a numpy ndarray
        of shape (num_states, num_actions).
      start_state_idx: Optional. Initial state of the environment. This will be
        the state that is first rendered.
    """
    self.mdp_env = mdp_env
    self.mdp_policy = mdp_policy
    tk.LabelFrame.__init__(master, text="Visualization")
    self.current_state_idx = start_state_idx
    self._build_gui()

  def _build_gui(self):
    """Builds the graphical user interface."""

    # Create an empty resizable grid of size 3x3.
    empty_image = tk.PhotoImage()
    for row_idx in range(self.mdp_env.num_rows):
      self.rowconfigure(row_idx, weight=1)
      for column_idx in range(self.mdp_env.num_columns):
        self.columnconfigure(column_idx, weight=1)
        xy_grid = tk.Label(
            master=self,
            width=100,
            height=100,
            background="white",
            borderwidth=2,
            relief="solid",
            image=empty_image,
        )
        xy_grid.grid(
            row=row_idx,
            column=column_idx,
            padx=0.1,
            pady=0.1,
            sticky="NSEW",
        )

    # Add robot to the grid.
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
    )
    self.robot_png_original = Image.open(os.path.join(data_path, "robot.png"))
    self.robot_png = self.robot_png_original.resize((80, 80), Image.ANTIALIAS)
    self.robot_itk_image = itk.PhotoImage(self.robot_png)
    self.robot = tk.Label(
        self,
        width=50,
        height=50,
        background="white",
        borderwidth=2,
        relief="solid",
        image=self.robot_itk_image,
    )

    # Render start state.
    self.render_state()

    # Setup methods for resizing windows.
    self.bind("<Configure>", self.resize)

  def resize(self, event):
    """Helper function for resizing the environment."""
    resize_dim = min((self.robot.winfo_width(), self.robot.winfo_height()))
    if resize_dim < 5:
      return

    resize_dim = int(resize_dim * 0.8)
    self.robot_png = self.robot_png_original.resize(
        (resize_dim, resize_dim),
        Image.ANTIALIAS,
    )
    self.robot_itk_image = itk.PhotoImage(self.robot_png)
    self.robot.configure(image=self.robot_itk_image)

  def render_state(self, state_idx = None):
    """Renders an MDP state.
    Args:
      state_idx: Optional. Index of an MDP state.
    """
    if state_idx is None:
      state_idx = self.current_state_idx

    # Convert state index to state.
    factored_state = self.mdp_env.np_idx_to_state[state_idx]
    state_x = self.mdp_env.x_space.idx_to_state[factored_state[0]]
    state_y = self.mdp_env.y_space.idx_to_state[factored_state[1]]

    # Render the state and update the state index.
    self.robot.grid(row=state_x, column=state_y, sticky="NSEW")
    self.current_state_idx = state_idx

  def render_policy(
      self,
      mdp_policy = None,
      demonstration_length = 1,
  ):
    """Plays an MDP policy.
    Args:
      mdp_policy: Optional. A stochastic policy specifying as a numpy ndarray
        of shape (num_states, num_actions).
      demonstration_length: Optional. The number of timesteps for which to
        render the policy.
    """
    if mdp_policy is None:
      logging.info("Rendering default policy")
      mdp_policy = self.mdp_policy

    if mdp_policy is None:
      raise ValueError("No policy to render.")

    # Play policy for one step.
    state_idx = self.current_state_idx
    action_idx = np.random.choice(
        self.mdp_env.num_actions,
        p=mdp_policy[state_idx],
    )
    next_state_idx = self.mdp_env.transition(state_idx, action_idx)
    self.render_state(next_state_idx)

    # Play next policy steps until demonstration is complete.
    demonstration_length -= 1
    if demonstration_length > 0:
        self.after(1000,
                 wrapped_partial(
                     self.render_policy,
                     mdp_policy=mdp_policy,
                     demonstration_length=demonstration_length,
                 ))


def visualise_grid():

    def create_grid(event=None):
        w = c.winfo_width()  # Get current width of canvas
        h = c.winfo_height()  # Get current height of canvas
        c.delete('grid_line')  # Will only remove the grid_line

        # Creates all vertical lines at intervals of 100
        for i in range(0, w, 100):
            c.create_line([(i, 0), (i, h)], tag='grid_line')

        # Creates all horizontal lines at intervals of 100
        for i in range(0, h, 100):
            c.create_line([(0, i), (w, i)], tag='grid_line')

        c.create_rectangle(0, 100, 100, 200, fill="red")
        c.create_text(50, 150, text="Goal1", fill="black", font=('Helvetica 15 bold'))
        c.create_rectangle(900, 100, 1000, 200, fill="red")
        c.create_text(950, 150, text="Goal2", fill="black", font=('Helvetica 15 bold'))

        agent_loc = [325, 275, 375, 275, 350, 225]  # (x0, y0), (x1, y1) <- minimum 3 points
        c.create_polygon(agent_loc, fill='black')
        c.create_text(350, 285, text="Agent", fill="black", font=('Helvetica 12 bold'))

    root = tk.Tk()
    c = tk.Canvas(root, height=300, width=1000, bg='white')
    c.pack(fill=tk.BOTH, expand=True)
    c.bind('<Configure>', create_grid)
    # c.postscript(file="RoadWorld.eps")
    # from PIL import Image
    # img = Image.open("RoadWorld.eps")
    # img.save("RoadWorld.png", "png")
    root.mainloop()


if __name__ == "__main__":
    visualise_grid()