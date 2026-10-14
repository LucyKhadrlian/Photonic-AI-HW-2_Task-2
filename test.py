from focusing import run_focusing

print(run_focusing.invoke(dict(macropixel_size=4, focus_radius=1, background_radius=10,
                               n_regions=8, n_iterations=20000, n_flips=1)))