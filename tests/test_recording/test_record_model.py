"""Tests for the @record_model decorator."""

from unittest.mock import Mock, patch

from mesa.model import Model

from mesa_llm.recording.record_model import _attach_recorder_to_agents, record_model
from mesa_llm.recording.simulation_recorder import SimulationRecorder


class TestAttachRecorderToAgents:
    """Test the _attach_recorder_to_agents utility function."""

    def test_attach_recorder_to_agents_with_recorder_attribute(self):
        """Test attaching recorder to agents that have recorder attribute."""
        # Create mock agents
        agent1 = Mock()
        agent1.recorder = None

        agent2 = Mock()
        agent2.recorder = None

        # Agent without recorder attribute
        agent3 = Mock()
        del agent3.recorder

        # Create mock model with proper agents list
        model = Mock()
        model.agents = [agent1, agent2, agent3]

        # Create mock recorder
        recorder = Mock(spec=SimulationRecorder)

        # Attach recorder
        _attach_recorder_to_agents(model, recorder)

        # Check that agents with recorder attribute got the recorder
        assert agent1.recorder == recorder
        assert agent2.recorder == recorder
        # agent3 should not have recorder attribute set
        assert not hasattr(agent3, "recorder")


class TestRecordModelDecorator:
    """Test the @record_model decorator with simplified scenarios."""

    def test_decorator_adds_recorder(self, temp_dir):
        """Test that decorator adds a recorder to the model."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        # Check that recorder was attached
        assert hasattr(model, "recorder")
        assert isinstance(model.recorder, SimulationRecorder)
        assert model.recorder.model == model
        assert model.recorder.output_dir == temp_dir

    def test_save_recording_method_added(self, temp_dir):
        """Test that save_recording method is added to the model."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        assert hasattr(model, "save_recording")
        assert callable(model.save_recording)

    def test_save_recording_method_works(self, temp_dir):
        """Test that save_recording method works correctly."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        # Mock the recorder's save method
        model.recorder.save = Mock(return_value="test_path.json")

        # Call save_recording
        result = model.save_recording(filename="test.json", format="json")

        # Check that recorder.save was called with correct arguments
        model.recorder.save.assert_called_once_with(filename="test.json", format="json")
        assert result == "test_path.json"

    def test_step_method_wrapping_basic(self, temp_dir):
        """Test basic step method wrapping functionality."""
        step_called = False

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                # Mesa might initialize steps to something other than 0
                self.test_steps = 0

            def step(self):
                nonlocal step_called
                step_called = True
                self.test_steps += 1

        model = SimpleModel()

        # Mock the recorder to track calls
        model.recorder = Mock(spec=SimulationRecorder)
        model.recorder.events = []  # instance attr not in class spec; set explicitly

        # Call step
        model.step()

        assert step_called
        assert model.test_steps == 1

        # Check that recorder methods were called
        assert model.recorder.record_model_event.call_count == 2

    def test_model_without_step_method(self, temp_dir):
        """Test that decorator works with models that don't have step method."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        # Should work without errors
        assert hasattr(model, "recorder")
        assert isinstance(model.recorder, SimulationRecorder)

    def test_decorator_with_kwargs(self, temp_dir):
        """Test decorator with various keyword arguments."""

        @record_model(
            output_dir=str(temp_dir), auto_save_interval=5, record_state_changes=False
        )
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        assert model.recorder.output_dir == temp_dir
        assert model.recorder.auto_save_interval == 5
        assert model.recorder.record_state_changes is False

    def test_original_init_called(self, temp_dir):
        """Test that original __init__ method is still called."""
        init_called = False

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                nonlocal init_called
                super().__init__()
                init_called = True
                self.steps = 0
                self.custom_attr = "test_value"

        model = SimpleModel()

        assert init_called
        assert model.custom_attr == "test_value"
        assert hasattr(model, "recorder")

    def test_class_attributes_preserved(self, temp_dir):
        """Test that decorator preserves original class attributes."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            """Test model docstring."""

            class_attr = "test_value"

            def __init__(self):
                super().__init__()
                self.steps = 0

            def custom_method(self):
                return "custom_result"

        # Check that class attributes are preserved
        assert SimpleModel.__doc__ == "Test model docstring."
        assert SimpleModel.class_attr == "test_value"

        model = SimpleModel()
        assert model.custom_method() == "custom_result"
        assert hasattr(model, "recorder")

    def test_step_method_execution(self, temp_dir):
        """Test that step method is executed correctly."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.test_steps = 0

            def step(self):
                self.test_steps += 1
                # Note: Mesa's Model.step() always returns None, so we focus on side effects

        model = SimpleModel()

        # Mock the record_model_event method to avoid file system issues
        model.recorder.record_model_event = Mock()

        result = model.step()

        # Mesa's step always returns None, but our custom logic should execute
        assert result is None  # This is expected behavior with Mesa
        assert model.test_steps == 1  # Our custom logic executed

    @patch("mesa_llm.recording.record_model.atexit.register")
    def test_atexit_registration(self, mock_atexit, temp_dir):
        """Test that auto-save is registered with atexit."""

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        SimpleModel()

        # Check that atexit.register was called
        mock_atexit.assert_called_once()

    def test_multiple_model_classes_independent(self, temp_dir):
        """Test that decorating multiple model classes works independently."""

        @record_model(output_dir=str(temp_dir / "model1"))
        class SimpleModel1(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        @record_model(output_dir=str(temp_dir / "model2"))
        class SimpleModel2(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model1 = SimpleModel1()
        model2 = SimpleModel2()

        # Each should have their own recorder
        assert model1.recorder != model2.recorder
        assert model1.recorder.output_dir == temp_dir / "model1"
        assert model2.recorder.output_dir == temp_dir / "model2"

    @patch("mesa_llm.recording.record_model.atexit.register")
    def test_multiple_instances_each_get_one_atexit_handler(
        self, mock_atexit, temp_dir
    ):
        """Test that creating N model instances registers exactly N atexit
        handlers — one per instance, not N for the first instance alone.

        Regression test for GitHub issue #202.
        """

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        SimpleModel()
        SimpleModel()
        SimpleModel()

        # Exactly 3 handlers — one per distinct model instance
        assert mock_atexit.call_count == 3

    @patch("mesa_llm.recording.record_model.atexit.register")
    def test_atexit_not_reregistered_on_reinit(self, mock_atexit, temp_dir):
        """Test that calling __init__ again on the same instance does not
        register another atexit handler.

        Regression test for GitHub issue #202.
        """

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()
        assert mock_atexit.call_count == 1

        # Re-initialize the same instance — should NOT add another handler
        model.__init__()
        assert mock_atexit.call_count == 1

    def test_atexit_registered_flag_set_on_instance(self, temp_dir):
        """Test that the _atexit_registered flag is set on each model instance
        after initialization.

        Regression test for GitHub issue #202.
        """

        @record_model(output_dir=str(temp_dir))
        class SimpleModel(Model):
            def __init__(self):
                super().__init__()
                self.steps = 0

        model = SimpleModel()

        assert hasattr(model, "_atexit_registered")
        assert model._atexit_registered is True
