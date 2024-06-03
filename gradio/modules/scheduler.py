from diffusers import LMSDiscreteScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler

class Schedulers():
    def lms():
        lms = LMSDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        return lms

    def dpm():
        dpm = DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        return dpm

    def euler():
        euler = EulerAncestralDiscreteScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear"
        )
        return euler