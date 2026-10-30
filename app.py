import datetime
import io
import os
import zipfile
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import numpy as np
import pandas as pd
import requests

# Mapeamento oficial das estações
ESTACOES = {
    "BRM00083899 - Florianopolis (SC)": "BRM00083899",
    "BRM00083840 - Curitiba (PR)": "BRM00083840",
    "BRM00083827 - Foz do Iguacu (PR)": "BRM00083827",
    "BRM00083768 - Londrina (PR)": "BRM00083768",
    "BRM00083937 - Santa Maria (RS)": "BRM00083937",
    "BRM00083612 - Campo Grande (MS)": "BRM00083612",
    "BRM00083779 - Sao Paulo/Marte (SP)": "BRM00083779",
    "ARM00087585 - Buenos Aires (ARG)": "ARM00087585",
    "ARM00087155 - Resistencia (ARG)": "ARM00087155",
}


def obter_perfil_igra(
    id_estacao: str, ano: int, mes: int, dia: int, hora: int
) -> pd.DataFrame:
  ano_atual = datetime.datetime.now().year
  diretorio = "data-y2d" if ano == ano_atual else "data-por"
  nome_base = (
      f"{id_estacao}-data-beg{ano_atual}.txt"
      if ano == ano_atual
      else f"{id_estacao}-data.txt"
  )
  url = (
      f"https://www.ncei.noaa.gov/pub/data/igra/data/{diretorio}/{nome_base}.zip"
  )

  headers = {
      "User-Agent": (
          "ProjetoRadiossondagem-AgulhaMagnetica/1.0"
          " (contato@agulhamagnetica.com)"
      )
  }

  try:
    resposta = requests.get(url, headers=headers, timeout=20)
    resposta.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resposta.content)) as z:
      nome_interno = z.namelist()[0]
      with z.open(nome_interno) as f:
        texto_bruto = f.read().decode("utf-8")
  except Exception:
    return pd.DataFrame()

  linhas = texto_bruto.splitlines()
  registros = []
  coletando = False
  cabecalho_alvo = f"#{id_estacao} {ano:04d} {mes:02d} {dia:02d} {hora:02d}"

  for linha in linhas:
    if linha.startswith("#"):
      if cabecalho_alvo in linha:
        coletando = True
        continue
      else:
        if coletando:
          break
        coletando = False
        continue

    if coletando:
      try:
        pressao_str = linha[9:15].strip()
        temp_str = linha[22:27].strip()
        dpdp_str = linha[34:39].strip()
        wdir_str = linha[40:45].strip()
        wspd_str = linha[46:51].strip()

        pressao = (
            float(pressao_str) / 100.0 if pressao_str != "-9999" else np.nan
        )
        temp = float(temp_str) / 10.0 if temp_str != "-9999" else np.nan
        dpdp = float(dpdp_str) / 10.0 if dpdp_str != "-9999" else np.nan
        direcao = float(wdir_str) if wdir_str != "-9999" else np.nan
        velocidade = (
            float(wspd_str) / 10.0 if wspd_str != "-9999" else np.nan
        )

        dewpoint = temp - dpdp if pd.notna(temp) and pd.notna(dpdp) else np.nan
        registros.append([pressao, temp, dewpoint, direcao, velocidade])
      except ValueError:
        continue

  df_temp = pd.DataFrame(
      registros,
      columns=["pressure", "temperature", "dewpoint", "direction", "speed"],
  )
  df_temp = df_temp.dropna(subset=["pressure", "temperature", "dewpoint"])
  df_temp = (
      df_temp.drop_duplicates(subset="pressure")
      .sort_values("pressure", ascending=False)
      .reset_index(drop=True)
  )
  return df_temp


def gerar_diagrama(estacao_nome, data_input, hora_z):
  id_estacao = ESTACOES[estacao_nome]
  local_nome = estacao_nome.split(" - ")[1]

  try:
    data_obj = datetime.datetime.strptime(data_input, "%Y-%m-%d")
  except ValueError:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "Formato de data inválido. Use AAAA-MM-DD.",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
    )
    ax.axis("off")
    return fig

  df = obter_perfil_igra(
      id_estacao,
      data_obj.year,
      data_obj.month,
      data_obj.day,
      int(hora_z.replace("Z", "")),
  )

  if df.empty:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(
        0.5,
        0.5,
        "Sondagem não disponível no repositório IGRA para esta data/hora.",
        ha="center",
        va="center",
        fontsize=12,
        color="red",
    )
    ax.axis("off")
    return fig

  p = df["pressure"].values * units.hPa
  T = df["temperature"].values * units.degC
  Td = df["dewpoint"].values * units.degC
  perfil = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")

  cape, cine = mpcalc.cape_cin(p, T, Td, perfil)
  p_ncl, T_ncl = mpcalc.lcl(p[0], T[0], Td[0])

  fig = plt.figure(figsize=(12, 8), layout="constrained")
  skew = SkewT(fig, rotation=45)

  skew.plot(p, T, color="black", linewidth=2)
  skew.plot(p, Td, color="darkblue", linewidth=1.5)
  skew.plot(p, perfil, color="black", linewidth=1.5, linestyle="--")

  skew.shade_cape(p, T, perfil, color="red", alpha=0.3)
  skew.shade_cin(p, T, perfil, Td, color="blue", alpha=0.3)

  skew.ax.set_xlim(-40, 40)
  skew.ax.set_ylim(1050, 100)
  skew.plot_dry_adiabats(alpha=0.25)
  skew.plot_moist_adiabats(alpha=0.25)

  w_valores = (
      np.array(
          [0.0002, 0.0004, 0.001, 0.002, 0.004, 0.007, 0.010, 0.016, 0.024]
      )
      * units("kg/kg")
  )
  skew.plot_mixing_lines(
      mixing_ratio=w_valores, alpha=0.3, colors="darkgreen", linestyles="dotted"
  )

  skew.ax.set_xlabel("Temperatura (°C)")
  skew.ax.set_ylabel("Pressão (hPa)")
  skew.ax.set_title(
      f"Estação: {id_estacao} - {local_nome}\nData:"
      f" {data_obj.strftime('%d/%m/%Y')} {hora_z} | Fonte: NOAA IGRA v2\nCAPE:"
      f" {cape.m:.0f} J/kg | CINE: {cine.m:.0f} J/kg | NCL: {p_ncl.m:.0f} hPa"
  )

  return fig


with gr.Blocks(title="Perfil Vertical da Atmosfera - Skew-T") as app:
  gr.Markdown("### Análise Termodinâmica de Radiossondagem (Skew-T Log-P)")
  gr.Markdown(
      "Selecione a estação, data e horário para traçar o perfil a partir dos"
      " dados do NOAA IGRA v2."
  )

  with gr.Row():
    estacao = gr.Dropdown(
        choices=list(ESTACOES.keys()),
        value="BRM00083899 - Florianopolis (SC)",
        label="Estação Meteorológica",
    )
    data = gr.Textbox(
        value=datetime.date.today().strftime("%Y-%m-%d"),
        label="Data (AAAA-MM-DD)",
    )
    hora = gr.Radio(choices=["00Z", "12Z"], value="12Z", label="Hora Sinótica")

  btn = gr.Button("Processar e Gerar Diagrama", variant="primary")
  grafico = gr.Plot(label="Diagrama Skew-T Log-P")

  btn.click(
      fn=gerar_diagrama, inputs=[estacao, data, hora], outputs=[grafico]
  )

# Executa na porta especificada pelo Render
porta = int(os.environ.get("PORT", 7860))
app.launch(server_name="0.0.0.0", server_port=porta)
