"""Nomes de exibição; os rótulos originais continuam na resposta da API."""

PLANTS = {
    "apple": "Macieira",
    "blueberry": "Mirtilo",
    "cherry_(including_sour)": "Cerejeira",
    "corn_(maize)": "Milho",
    "grape": "Videira",
    "orange": "Laranjeira",
    "peach": "Pessegueiro",
    "pepper,_bell": "Pimentão",
    "potato": "Batata",
    "raspberry": "Framboesa",
    "soybean": "Soja",
    "squash": "Abóbora",
    "strawberry": "Morango",
    "tomato": "Tomateiro",
}

CONDITIONS = {
    "healthy": "Saudável",
    "apple_scab": "Sarna da macieira",
    "black_rot": "Podridão negra",
    "cedar_apple_rust": "Ferrugem da macieira",
    "powdery_mildew": "Oídio",
    "cercospora_leaf_spot_gray_leaf_spot": "Cercosporiose",
    "common_rust": "Ferrugem comum",
    "northern_leaf_blight": "Helmintosporiose",
    "esca_(black_measles)": "Esca",
    "leaf_blight_(isariopsis_leaf_spot)": "Mancha foliar de Isariopsis",
    "haunglongbing_(citrus_greening)": "Greening (HLB)",
    "bacterial_spot": "Mancha bacteriana",
    "early_blight": "Pinta-preta",
    "late_blight": "Requeima",
    "leaf_scorch": "Queima das folhas",
    "leaf_mold": "Mofo das folhas",
    "septoria_leaf_spot": "Septoriose",
    "spider_mites_two_spotted_spider_mite": "Ácaro-rajado",
    "target_spot": "Mancha-alvo",
    "tomato_yellow_leaf_curl_virus": "Vírus do amarelecimento e enrolamento das folhas",
    "tomato_mosaic_virus": "Vírus do mosaico do tomateiro",
}


def describe_class(label: str) -> dict[str, str]:
    normalized = label.lower().replace("___", "__").replace(" ", "_").replace("-", "_")
    plant, separator, condition = normalized.partition("__")
    condition = condition.strip("_")
    return {
        "class_name": label,
        "plant": PLANTS.get(plant, plant.replace("_", " ").capitalize()),
        "condition": CONDITIONS.get(
            condition, condition.replace("_", " ").capitalize() if separator else label
        ),
    }
