from kombu import Queue

celery_app.conf.update(
    task_queues=[
        Queue('transcribe_cpu'),
        Queue('transcribe_gpu'),
        Queue('diarize_gpu'),
    ],
    task_routes={
        'tasks.preview_slice':      {'queue': 'transcribe_cpu'},
        'tasks.preview_whisper':    {'queue': 'transcribe_gpu'},
        'tasks.transcribe_segments':{'queue': 'transcribe_gpu'},
        'tasks.diarize_full':       {'queue': 'diarize_gpu'},
    },
)