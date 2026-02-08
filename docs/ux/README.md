\# Airline Predict — UX/UI Decisions



This folder contains the UX/UI design rationale for the Airline Predict Streamlit application.



\## Purpose



Translate the ML passenger satisfaction model into a usable product interface for Airline Customer Experience teams.



The goal is to reduce cognitive load, improve input quality, and make prediction outputs understandable for non-technical users.



\## Key UX Principles Applied



\- Low cognitive load form structure

\- Inputs grouped by journey moments

\- CSAT Likert 1–5 semantic scale

\- Clear labels and microcopy

\- Accessible input controls

\- Expandable sections instead of long forms

\- Visible progress and feedback

\- ML prediction panel with confidence + drivers



\## CSAT Handling Rule



Service rating variables use a 1–5 Likert scale.



Value 0 in the dataset is treated as:

Not rated / Not applicable / Service not used



In modeling:

\- recoded as missing (NaN)

\- optional binary “service\_used” flags created



\## Target User



Airline Customer Experience Manager  

B2B analytical workflow  

Decision-support tool — not a consumer app



\## Files



\- ui-ux-decisions-airline-predict.pdf → full UX/UI design document

